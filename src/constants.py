TARGET_URLS = [
    "https://help.typeform.com/hc/en-us/articles/23541138531732-Create-multi-language-forms", # provided
    "https://help.typeform.com/hc/en-us/articles/27703634781076-Add-a-Multi-Question-Page-to-your-form", # provided
    "https://help.typeform.com/hc/en-us/articles/14955071444244-Use-AI-to-create-forms",
    "https://help.typeform.com/hc/en-us/articles/360029294452-Protect-access-to-your-form",
    "https://help.typeform.com/hc/en-us/articles/360029294412-Use-a-template",
    "https://help.typeform.com/hc/en-us/articles/360054384092-Create-a-quiz",
    "https://help.typeform.com/hc/en-us/articles/36047433913364-Which-type-of-Logic-should-I-use-and-where-can-I-find-it",
    "https://help.typeform.com/hc/en-us/articles/360029116392-What-is-Logic",
    "https://help.typeform.com/hc/en-us/articles/360057591531-Logic-Map",
    "https://help.typeform.com/hc/en-us/articles/4403183643796-Ordering-Logic",
    "https://help.typeform.com/hc/en-us/articles/360052588072-Get-the-most-out-of-Logic-variables-and-scores-while-ordering-pizza",
    "https://help.typeform.com/hc/en-us/articles/4403183498644-Using-and-or-with-Logic",
    "https://help.typeform.com/hc/en-us/articles/23794452273812-Data-enrichment-with-Typeform",
    "https://help.typeform.com/hc/en-us/articles/32764583548564-Values-for-Data-enrichment-variable-enrich-country",
    "https://help.typeform.com/hc/en-us/articles/35948010629396-Advanced-spam-protection-with-invisible-bot-detection",
    "https://help.typeform.com/hc/en-us/articles/27917825492244-Prevent-duplicate-responses",
    "https://help.typeform.com/hc/en-us/articles/25762929948692-Add-video-questions-to-your-form",
    "https://help.typeform.com/hc/en-us/articles/360052429631-Add-images-and-GIFs-to-your-forms",
    "https://help.typeform.com/hc/en-us/articles/360029256192-How-to-set-up-the-Typeform-Google-Sheets-integration",
    "https://help.typeform.com/hc/en-us/articles/360029251312-Integrate-your-typeform-with-Google-Analytics",
    "https://help.typeform.com/hc/en-us/articles/360042732171-Salesforce-integration-Installation-and-setup",
    "https://help.typeform.com/hc/en-us/articles/4404606710420-Salesforce-integration-Mapping-and-records",
    "https://help.typeform.com/hc/en-us/articles/4404606654484-Salesforce-integration-FAQ",
    "https://help.typeform.com/hc/en-us/articles/360029253572-Export-your-responses",
    "https://help.typeform.com/hc/en-us/articles/360029253732-Working-with-your-responses",
    "https://help.typeform.com/hc/en-us/articles/23542072977172-Get-AI-analysis-of-your-results-with-Smart-Insights"
]

# Retrieval
INDEX_NAME = "help-centre-hybrid"
SENTENCE_TRANSFORMER = "all-MiniLM-L6-v2"
ALPHA = 0.5
RETRIEVAL_TOP_N = 10

# Reranking
RERANKER_TOP_N = 3
RERANKER_MODEL = "bge-reranker-v2-m3"

# LLM
ASSISTANT_MODEL = "gpt-4.1"

# Langchain
LANGCHAIN_PROJECT = "HELP_CENTRE_RAG"