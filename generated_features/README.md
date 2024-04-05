# File Structure for `generated_features` Directory
This directory contains the generated features for each context (background text) as JSON files.

Each JSON file corresponds to a context and follows the following structure:
```json
{
    "title": {{title of the context}},
    {{feature name (e.g. 'summaries')}}: {
        {{ID of corresponding chunk in the context}}: {{generated text for the feature}}
    },
    {{other features}}
}
```