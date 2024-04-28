import json
import os
import time
import re

from llama_index.core import Settings, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from datasets import load_dataset
import evaluate
import tiktoken
import numpy as np


from tqdm.auto import tqdm

from models import mixtral, mistral_large, gpt4


def count_tokens(string):
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_ids_from_index(index):
    return list(index.vector_store.to_dict()["embedding_dict"].keys())


def get_nodes_from_index(index):
    ids = get_ids_from_index(index)
    nodes = index.docstore.get_nodes(ids)
    return nodes


def get_node_by_id(index, id_):
    return index.docstore.get_node(id_)


def get_text_by_id(index, id_):
    return get_node_by_id(index, id_).text


def similarity_score(query, index_id, node_id):
    index = get_index_by_title(index_id)
    node_embedding = np.array(index.vector_store.get(node_id))
    query_embedding = np.array(Settings.embed_model._get_query_embedding(query))
    score = np.dot(node_embedding, query_embedding)
    return score


def chunk_text(text, chunk_size=1024, chunk_overlap=200):
    """
    Split the document text into chunks using the default "sentence splitter", which favors splitting at sentence boundaries.

    Args:
        text (str): The entire document text.
        chunk_size (int): The maximum number of characters in each chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        list[str]: A list of text chunks.
    """
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return chunks


def get_index_by_title(title):
    """
    Get an index by its title. If the index does not exist, returns None.

    Args:
        title (str): The title of the index
    Returns:
        VectorStoreIndex: The index with the specified title
    """
    try:
        storage_context = StorageContext.from_defaults(persist_dir=os.path.join("indices", title))
        index = load_index_from_storage(storage_context, index_id=title)
        return index
    except:
        return None


def create_index_from_chunks_with_ids(chunks, ids, index_title, overwrite_existing=False):
    index = get_index_by_title(index_title)

    if index is None or overwrite_existing:
        chunk_nodes = [TextNode(id_=id_, text=chunk) for id_, chunk in zip(ids, chunks)]
        index = VectorStoreIndex(chunk_nodes)
        index.set_index_id(index_title)
        storage_dir = os.path.join("indices", index_title)
        os.makedirs(storage_dir, exist_ok=True)
        index.storage_context.persist(persist_dir=storage_dir)

    return index


def create_index_from_text_with_ids(text, index_title, chunk_size=1024, chunk_overlap=200):
    """
    Create an index from context, assigning a unique ID to each text chunk without including the ID in the text used for embedding.

    Args:
        text (str): The context to index.
        index_title (str): The title of the index for access later.
        chunk_size (int): The maximum number of characters in each chunk.

    Returns:
        VectorStoreIndex: The index created from the context, with IDs assigned to each text chunk.
    """
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    index = get_index_by_title(index_title)

    if index is None:
        print(f"Creating index {index_title} from text.")
        ids = [f"text_chunk_{i}" for i in range(len(chunks))]
        index = create_index_from_chunks_with_ids(chunks, ids, index_title)

    return index


def index_union(indices):
    """
    Create a new index by combining the nodes of multiple indices.

    Args:
        indices (list[VectorStoreIndex]): The indices to combine.

    Returns:
        VectorStoreIndex: The index containing all nodes from the input indices.
    """
    combined_nodes = []
    for index in indices:
        nodes = get_nodes_from_index(index)
        combined_nodes.extend(nodes)

    combined_index = VectorStoreIndex(combined_nodes)
    return combined_index


longdep_qa_ds = load_dataset("bigainlco/LooGLE", "longdep_qa", split="test")
rouge = evaluate.load("rouge")


def get_rouge_metrics(output_file):
    """
    Get ROUGE metrics for a .jsonl file containing generated answers and ground truth answers.

    Args:
        output_file (str): The path to the .jsonl file

    Returns:
        dict: The ROUGE metrics
    """
    with open(output_file, "r") as f:
        lines = f.readlines()
    outputs = [json.loads(line) for line in lines]
    generated_answers = [output["generated_answer"] for output in outputs]
    ground_truths = [output["ground_truth"] for output in outputs]
    rouge_metrics = rouge.compute(predictions=generated_answers, references=ground_truths)
    return rouge_metrics


def llm_self_score(output_file, llm=Settings.llm):
    """
    Score the generated answers in a .jsonl file using the LLM. The user will be prompted to determine whether each answer is correct.

    Args:
        output_file (str): The path to the .jsonl file

    Returns:
        float: The accuracy of the generated answers
    """
    with open(output_file, "r") as f:
        lines = f.readlines()
    outputs = [json.loads(line) for line in lines]
    for output in tqdm(outputs):
        if "correct" in output:
            # print("Skipping already scored output")
            continue
        question = output["question"]
        ground_truth = output["ground_truth"]
        generated_answer = output["generated_answer"]
        prompt = f"""\Question: {question}\n
            \Answer: {ground_truth}\n
            \Predicted Answer: {generated_answer}\n\n
            \given the question and the ground truth answer, is the predicted answer correct?\n
            \True or False?: """
        prompt = re.sub(r"\s+", " ", prompt)
        response = llm.complete(prompt).text.lower()
        if "false" in response:
            output["correct"] = False
        else:
            output["correct"] = True
    num_correct = sum([output["correct"] for output in outputs])
    accuracy = num_correct / len(outputs)
    with open(output_file, "w") as f:
        for output in outputs:
            json.dump(output, f)
            f.write("\n")
    return accuracy



def read_output_file(output_file):
    """
    Read a .jsonl file containing generated answers and ground truth answers.

    Args:
        output_file (str): The path to the .jsonl file

    Returns:
        list: The outputs in the file
    """
    if not os.path.exists(output_file):
        return []
    with open(output_file, "r") as f:
        lines = f.readlines()
    outputs = [json.loads(line) for line in lines]
    return outputs


def log_outputs(question, ground_truth, generated_answer, additional_info, output_file):
    """
    Log a question, its ground truth answer, and a generated answer to a .jsonl file.

    Args:
        question (str): The question
        ground_truth (str): The ground truth answer
        generated_answer (str): The generated answer
        additional_info (dict): Additional information to log
        output_file (str): The path to the .jsonl file; if None, a new file will be created in the "output" directory with the current time as the name

    Returns:
        tuple: The path to the .jsonl file and the outputs in the file
    """
    if output_file is None:
        os.makedirs("output", exist_ok=True)
        output_file = f"output/{time.time()}.jsonl"
    with open(output_file, "a") as f:
        json.dump(
            {
                "question": question,
                "ground_truth": ground_truth,
                "generated_answer": generated_answer,
                "additional_info": additional_info,
            },
            f,
        )
        f.write("\n")
    existing_output = read_output_file(output_file)
    return output_file, existing_output


def question_is_answered(question, existing_output):
    """
    Determine whether a question has already been answered in a list of outputs.

    Args:
        question (str): The question
        existing_output (list): The outputs

    Returns:
        bool: Whether the question has already been answered
    """
    if question in [output["question"] for output in existing_output]:
        return True
    return False


def test_longdep_qa(
    inference_function, output_file=None, debug_lim=None, qa_llm=gpt4, chunk_size=1024, top_k=2, chunk_overlap=200
):
    """
    Test an inference function on the longdep_qa dataset.

    Args:
        inference_function (function): The function to test
        output_file (str): The path to the .jsonl file to log outputs to; if None, a new file will be created in the "output" directory with the current time as the name
        debug_lim (int): The number of questions to test; if None, all questions will be tested

    Returns:
        None
    """
    n_questions = sum([len(eval(env["qa_pairs"])) for env in longdep_qa_ds])
    if debug_lim is None:
        debug_lim = n_questions
    existing_output = read_output_file(output_file)
    with tqdm(total=debug_lim, position=0, desc="Answering questions") as pbar:
        for environment in longdep_qa_ds:
            context = environment["input"]
            title = environment["title"]
            title = f"{title}_chunksize{chunk_size}"
            qa_pairs = eval(environment["qa_pairs"])
            for question_dict in qa_pairs:
                question = question_dict["Q"]
                ground_truth = question_dict["A"]
                if not question_is_answered(question, existing_output):
                    inference_response = inference_function(
                        question,
                        context_title=title,
                        context_text=context,
                        qa_llm=qa_llm,
                        chunk_size=chunk_size,
                        top_k=top_k,
                        chunk_overlap=chunk_overlap,
                    )
                    if isinstance(inference_response, tuple):
                        generated_answer = inference_response[0]
                        additional_info = inference_response[1]
                    else:
                        generated_answer = inference_response
                        additional_info = {}
                    output_file, existing_output = log_outputs(
                        question, ground_truth, generated_answer, additional_info, output_file
                    )
                pbar.update(1)
                if pbar.n >= debug_lim:
                    break
            if pbar.n >= debug_lim:
                break


def generate_qa_prompt(top_chunks_text_combined, question):
    qa_prompt = f"""<s>[INST]Consider the following context with depth and thoughtfulness: {top_chunks_text_combined}\n\n\
        Respond to the following question with insight and nuance. Answer concisely, often in one \
        sentence or less and sometimes in the form of a list or structured text. If the question \
        asks you to order events, refer to the events by their number (e.g. "1. third event, 2. second \
        event, 3. first event" -> "3, 2, 1"). Answer multiple choice questions using the number which \
        corresponds to the correct answer (e.g. "1. A, 2. B, 3. C" -> "2"). Do not include the \
        question in your answer. \
        \n\n\
        Question: {question}\n\n [/INST]\
        Answer: """
    return qa_prompt

def generate_qa_prompt_icl(top_chunks_text_combined, question, example_list):
    qa_prompt = f"""Your task is to consider the context with depth and thoughtfulness and respond to the following question with insight and nuance.\n
        Answer the question to achieve a high score. To score ranges from 0 to 1.\n
        If the question asks you to order events, refer to the events by their number (e.g. "1. third event, 2. second \
        event, 3. first event" -> "3, 2, 1"). Answer multiple choice questions using the number which \
        corresponds to the correct answer (e.g. "1. A, 2. B, 3. C" -> "2"). Do not include the \
        question in your answer. \
        Below are some sample question answers.\n\n
        Question1: {example_list[0]['question']}\n\n
        Context: {example_list[0]['context']}\n\n
        Ground Truth: {example_list[0]['ground']}\n
        Generated Answer: {example_list[0]['generated']}\n
        Score: {example_list[0]['score']}\n\n\n
        Question2: {example_list[1]['question']}\n\n
        Context: {example_list[1]['context']}\n\n
        Ground Truth: {example_list[1]['ground']}\n
        Generated Answer: {example_list[1]['generated']}\n
        Score: {example_list[1]['score']}\n\n\n
        Question3: {example_list[2]['question']}\n\n
        Context: {example_list[2]['context']}\n\n
        Ground Truth: {example_list[2]['ground']}\n
        Generated Answer: {example_list[2]['generated']}\n
        Score: {example_list[2]['score']}\n\n\n
        Question4: {example_list[3]['question']}\n\n
        Context: {example_list[3]['context']}\n\n
        Ground Truth: {example_list[3]['ground']}\n
        Generated Answer: {example_list[3]['generated']}\n
        Score: {example_list[3]['score']}\n\n\n
        Question5: {example_list[4]['question']}\n\n
        Context: {example_list[4]['context']}\n\n
        Ground Truth: {example_list[4]['ground']}\n
        Generated Answer: {example_list[4]['generated']}\n
        Score: {example_list[4]['score']}\n\n\n
        \n\n\
        Based on these examples and instructions, generate a response to the following question.
        Question: {question}\n\n
        Context: {top_chunks_text_combined}\n\n[/INST]\
        Answer: """
    return qa_prompt

def answer_reading_comprehension(question, retrieved_chunks_combined, qa_llm=mistral_large):
    prompt = generate_qa_prompt(retrieved_chunks_combined, question)
    response = qa_llm.complete(prompt).text
    return response


def test_longdep_qa_icl(
    inference_function, output_file=None, debug_lim=None, qa_llm=mistral_large, chunk_size=1024, top_k=2, chunk_overlap=200
):
    """
    Test an inference function on the longdep_qa dataset.

    Args:
        inference_function (function): The function to test
        output_file (str): The path to the .jsonl file to log outputs to; if None, a new file will be created in the "output" directory with the current time as the name
        debug_lim (int): The number of questions to test; if None, all questions will be tested

    Returns:
        None
    """
    llm=Settings.llm
    n_questions = sum([len(eval(env["qa_pairs"])) for env in longdep_qa_ds])
    if debug_lim is None:
        debug_lim = n_questions
    existing_output = read_output_file(output_file)
    example_list = []
    temp_list = []
    with tqdm(total=debug_lim, position=0, desc="Answering questions") as pbar:
        for environment in longdep_qa_ds:
            context = environment["input"]
            title = environment["title"]
            title = f"{title}_chunksize{chunk_size}"
            qa_pairs = eval(environment["qa_pairs"])
            for question_dict in qa_pairs:
                question = question_dict["Q"]
                ground_truth = question_dict["A"]
                if not question_is_answered(question, existing_output):
                    inference_response = inference_function(
                        question,
                        context_title=title,
                        context_text=context,
                        qa_llm=qa_llm,
                        chunk_size=chunk_size,
                        top_k=top_k,
                        chunk_overlap=chunk_overlap,
                        examplelist=example_list
                    )
                    if isinstance(inference_response, tuple):
                        generated_answer = inference_response[0]
                        additional_info = inference_response[1]
                    else:
                        generated_answer = inference_response
                        additional_info = {}
                    output_file, existing_output = log_outputs(
                        question, ground_truth, generated_answer, additional_info, output_file
                    )
                    
                    prompt = f"""\Question: {question}\n
                        \Answer: {ground_truth}\n
                        \Predicted Answer: {generated_answer}\n\n
                        \given the question and the ground truth answer, is the predicted answer correct?\n
                        \Give a score between 0 and 1, where 1 is the perfect answer.Just return the score value.\n
                        \Score?: """
                    prompt = re.sub(r"\s+", " ", prompt)
                    response = llm.complete(prompt).text.lower()
                    #create dictionary for example_list
                    temp_list.append({'question':question, 'context':context, 'ground':ground_truth, 'generated':generated_answer, 'score':response})
                    if len(temp_list) == 5 and pbar.n <=105:
                        example_list = temp_list
                        temp_list = []
                pbar.update(1)
                if pbar.n >= debug_lim:
                    break
            if pbar.n >= debug_lim:
                break