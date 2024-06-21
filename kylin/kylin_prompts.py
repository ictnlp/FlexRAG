determine_info_prompt = [
    {
        "role": "system",
        "content": (
            "Please indicate the external knowledge needed to answer the following question. "
            "If the question can be answered without external knowledge, "
            'answer "No additional information is required".\n\n'
        ),
    },
    {
        "role": "user",
        "content": "Question: who got the first nobel prize in physics.",
    },
    {
        "role": "assistant",
        "content": "[1] The first physics Nobel Laureate.",
    },
    {
        "role": "user",
        "content": "Question: big little lies season 2 how many episodes",
    },
    {
        "role": "assistant",
        "content": "[1] The total number of episodes in the first season.",
    },
    {
        "role": "user",
        "content": "Question: who sings i can't stop this feeling anymore?",
    },
    {
        "role": "assistant",
        "content": '[1] The singer of "i can\'t stop this feeling anymore".',
    },
    {
        "role": "user",
        "content": "what is 1 radian in terms of pi?",
    },
    {
        "role": "assistant",
        "content": "No additional information is required.",
    },
    {
        "role": "user",
        "content": "Question: Which magazine was started first Arthur's Magazine or First for Women?",
    },
    {
        "role": "assistant",
        "content": (
            "[1] The time when Arthur's Magazine was founded.\n"
            "[2] The time when First for Women was founded."
        ),
    },
]


# TODO: few-shot learning
rewrite_prompts = {
    "bm25": [
        {
            "role": "system",
            "content": (
                "Please optimize the following query for the BM25 search engine.\n"
                "Suggestions for Writing Queries for BM25 Search Engine\n"
                "1. Use Descriptive Keywords: Ensure your query includes all relevant keywords that describe what you are searching for.\n"
                "2. Incorporate Rare Terms: If you know any specific or rare terms related to your search, include them.\n"
                '3. Avoid Stop Words: Common words like "the", "is", and "and" may dilute the effectiveness of the query.\n'
                "4. Phrase Searches: When searching for specific phrases, enclose them in single quotes.\n"
                "5. Synonyms and Related Terms: Use synonyms and related terms to cover variations in how different documents might reference the same concept.\n"
                "6. Use Boolean Operators: Use double quotes for terms that must contains in the documents, and minus to exclude terms in your query.\n"
                'For example, to search for documents about MacBook Air that do not mention iPhones, you could use the query: "MacBook Air" Apple  -iPhone\n'
                "Please only reply your query and do not output any other words."
            ),
        }
    ],
    "dense": [
        {
            "role": "system",
            "content": (
                "Please optimize the following query for the BM25 search engine.\n"
                "Suggestions for Writing Queries for Dense Retrieval Search Engine\n"
                "1. Use Natural Language: Dense retrieval models are designed to understand and process natural language. Formulate queries in complete sentences or phrases as you would ask a question in a conversation.\n"
                "2. Incorporate Context: Provide context to your query to help the model understand the specific aspect of the topic you are interested in. Contextual information improves the accuracy of the retrieval.\n"
                "3. Ask Specific Questions: Dense retrieval models perform well with specific queries. Instead of a single keyword, use detailed questions or statements to convey your information need.\n"
                "4. Avoid Overly Technical Language: While dense retrieval can handle a variety of terms, overly technical or jargon-heavy language might not be necessary. Aim for clarity and simplicity.\n"
                "Please only reply your query and do not output any other words."
            ),
        }
    ],
    "web": [
        {
            "role": "system",
            "content": (
                "Please optimize the following query for the BM25 search engine.\n"
                "Suggestions for Writing Queries for Web Search Engines\n"
                "1. Use Specific Keywords: Identify and use the most relevant and specific keywords related to your search topic. This helps narrow down the results to the most pertinent web pages.\n"
                "2. Phrase Searches: Enclose exact phrases in quotation marks to search for those exact words in that exact order. This is useful for finding specific quotes, names, or titles.\n"
                '3. Ask Questions: Formulate your query as a question to get direct answers. For example, "How to cook pasta?" is likely to return step-by-step instructions.\n'
                '4. Synonyms and Variants: Include synonyms or different variations of a word to broaden your search. For instance, "smartphone" and "mobile phone" can yield different results.\n'
                "Please only reply your query and do not output any other words."
            ),
        }
    ],
}


# TODO: few-shot learning
verify_prompt = [
    {
        "role": "system",
        "content": (
            "Your task is to verify whether the following context "
            "contains enough information to understand the following topic. "
            "Please only reply 'yes' or 'no' and do not output any other words."
        ),
    }
]


# TODO: few-shot learning
feedback_prompt = [
    {
        "role": "system",
        "content": (
            "If the context is completely irrelevant to the question, "
            "please reply with the keywords that should be filtered out to avoid retrieving the same information in the next search, "
            "or reply with the keywords that should be contained in the next search."
        ),
    }
]


summary_prompt = [
    {
        "role": "system",
        "content": (
            "Please copy the sentences from the following context that are relevant to the given question.\n"
            "If the context contains relevant sentences, please only reply the relevant sentences and do not output any other words.\n"
            "If the context does not contain any relevant information, please reply 'No relevant information' and do not output any other words."
        ),
    },
    {
        "role": "user",
        "content": (
            "Context: Kaieteur Falls  waterfalls in the world with an average flow rate of 663 cubic metres per second (23,400 cubic feet per second). "
            "Kaieteur Falls is about four and a half times the height of Niagara Falls, on the border between Canada and the United States, "
            "and about twice the height of Victoria Falls, on the border of Zambia and Zimbabwe in Africa. "
            "It is a single drop waterfall. Upriver from the falls, the Potaro Plateau stretches out to the distant escarpment of the Pakaraima Mountains. "
            "The Potaro River empties into the Essequibo River which is one of the longest and widest rivers in South America and the longest river in Guyana.\n\n"
            "Question: Which is the second longest river in Africa?"
        ),
    },
    {
        "role": "assistant",
        "content": "No relevant information",
    },
    {
        "role": "user",
        "content": (
            "Context: Congo River  The Congo River (Nzâdi Kôngo, Fleuve Congo, Rio Congo), "
            "formerly also known as the Zaire River, is the second longest river in Africa, "
            "shorter only than the Nile, as well as the second largest river in the world by discharge volume, "
            "following only the Amazon. "
            "It is also the world's deepest recorded river, with measured depths in excess of 220 m. "
            "The Congo-Lualaba-Chambeshi River system has an overall length of 4700 km, which makes it the world's ninth-longest river. "
            "The Chambeshi is a tributary of the Lualaba River, and Lualaba is the name of the Congo River upstream of Boyoma Falls, extending for 1800 km. "
            "Measured along with the Lualaba, the main tributary, the Congo River has a total length of 4370 km. "
            "It is the only major river to cross the equator twice. The Congo Basin has a total area of about 4000000 km2, or 13% of the entire African landmass.\n\n"
            "Question: Which is the second longest river in Africa?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Congo River  The Congo River (Nzâdi Kôngo, Fleuve Congo, Rio Congo), "
            "formerly also known as the Zaire River, is the second longest river in Africa, "
            "shorter only than the Nile, as well as the second largest river in the world by discharge volume, following only the Amazon. "
        ),
    },
]


# TODO: few-shot learning
generate_prompt = [
    {
        "role": "system",
        "content": (
            "Answer the question based on the given contexts. "
            "Note that the context might not always contain relevant information to answer the question. "
            "Only give me the answer and do not output any other words."
        ),
    }
]
