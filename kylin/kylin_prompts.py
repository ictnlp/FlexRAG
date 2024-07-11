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


rewrite_prompts = {
    "bm25": [
        {
            "role": "system",
            "content": (
                "Suggestions for Writing Queries for BM25 Search Engine\n"
                "1. Use Descriptive Keywords: Ensure your query includes all relevant keywords that describe what you are searching for.\n"
                "2. Incorporate Rare Terms: If you know any specific or rare terms related to your search, include them.\n"
                '3. Avoid Stop Words: Common words like "the", "is", and "and" may dilute the effectiveness of the query.\n'
                "4. Synonyms and Related Terms: Use synonyms and related terms to cover variations in how different documents might reference the same concept.\n"
                "5. Entity Searches: When searching for specific named entity, enclose them in double quotes.\n"
                "Please optimize the following query for the BM25 Search Engine.\n"
                "Please only reply your query and do not output any other words."
            ),
        },
        # case from popqa
        {
            "role": "user",
            "content": "What is John Mayne's occupation?",
        },
        {
            "role": "assistant",
            "content": '"John Mayne" occupation job career',
        },
        # case from nq
        {
            "role": "user",
            "content": "how many oar athletes are in the olympics",
        },
        {
            "role": "assistant",
            "content": '"oar athletes" olympics number count participants',
        },
        {
            "role": "user",
            "content": "who introduced the system of civil services in india",
        },
        {
            "role": "assistant",
            "content": "india civil services introduced foundation",
        },
        # case from triviaqa
        {
            "role": "user",
            "content": "Which leader did Hitler meet in the Brenner Pass in WWII?",
        },
        {
            "role": "assistant",
            "content": 'Hitler "Brenner Pass" WWII leader meeting',
        },
        {
            "role": "user",
            "content": "Which country does the airline Garuda come from?",
        },
        {
            "role": "assistant",
            "content": "Garuda airline country origin",
        },
    ],
    "bm25_advance": [
        {
            "role": "system",
            "content": (
                "Suggestions for Writing Queries for BM25 Search Engine\n"
                "1. Use Descriptive Keywords: Ensure your query includes all relevant keywords that describe what you are searching for.\n"
                "2. Incorporate Rare Terms: If you know any specific or rare terms related to your search, include them.\n"
                '3. Avoid Stop Words: Common words like "the", "is", and "and" may dilute the effectiveness of the query.\n'
                "4. Synonyms and Related Terms: Use synonyms and related terms to cover variations in how different documents might reference the same concept.\n"
                "5. Phrase Searches: When searching for specific named entity, enclose them in double quotes.\n"
                '6. Use Boolean Operators: Use "+" for terms that must contains in the documents, "-" for terms that must not contains in the documents, and "OR" for terms that are optional.\n'
                "Please optimize the following query for the BM25 Search Engine.\n"
                "Please only reply your query and do not output any other words."
            ),
        },
        {
            "role": "user",
            "content": "What is John Mayne's occupation?",
        },
        {
            "role": "assistant",
            "content": '+"John Mayne" (occupation OR job OR career OR profession)',
        },
        {
            "role": "user",
            "content": "what year did my pet monster come out",
        },
        {
            "role": "assistant",
            "content": '+"my pet monster" (year OR release OR debut OR came)',
        },
        {
            "role": "user",
            "content": "Which prince is Queen Elizabeth II's youngest son?",
        },
        {
            "role": "assistant",
            "content": '+"Queen Elizabeth II" +youngest +(prince OR son OR child)',
        },
    ],
    "dense": [
        {
            "role": "system",
            "content": (
                "Suggestions for Writing Queries for Dense Retrieval Search Engine\n"
                "1. Use Natural Language: Dense retrieval models are designed to understand and process natural language. Formulate queries in complete sentences or phrases as you would ask a question in a conversation.\n"
                "2. Incorporate Context: Provide context to your query to help the model understand the specific aspect of the topic you are interested in. Contextual information improves the accuracy of the retrieval.\n"
                "3. Ask Specific Questions: Dense retrieval models perform well with specific queries. Instead of a single keyword, use detailed questions or statements to convey your information need.\n"
                "4. Avoid Overly Technical Language: While dense retrieval can handle a variety of terms, overly technical or jargon-heavy language might not be necessary. Aim for clarity and simplicity.\n"
                "Please optimize the following query for the Dense Retrieval Search Engine.\n"
                "Please only reply your query and do not output any other words."
            ),
        },
        {
            "role": "user",
            "content": "What is John Mayne's occupation?",
        },
        {
            "role": "assistant",
            "content": "What is John Mayne's occupation?",
        },
        {
            "role": "user",
            "content": "The year my pet monster come out",
        },
        {
            "role": "assistant",
            "content": "When did My Pet Monster come out?",
        },
    ],
    "web": [
        {
            "role": "system",
            "content": (
                "Please optimize the following query for the Web search engine.\n"
                "Suggestions for Writing Queries for Web Search Engines:\n"
                "1. Use Specific Keywords: Identify and use the most relevant and specific keywords related to your search topic. This helps narrow down the results to the most pertinent web pages.\n"
                "2. Phrase Searches: Enclose exact phrases in quotation marks to search for those exact words in that exact order. This is useful for finding specific quotes, names, or titles.\n"
                '3. Ask Questions: Formulate your query as a question to get direct answers. For example, "How to cook pasta?" is likely to return step-by-step instructions.\n'
                '4. Synonyms and Variants: Include synonyms or different variations of a word to broaden your search. For instance, "smartphone" and "mobile phone" can yield different results.\n'
                "Please only reply your query and do not output any other words."
            ),
        },
        {
            "role": "user",
            "content": "What is John Mayne's occupation?",
        },
        {
            "role": "assistant",
            "content": 'What is "John Mayne"\'s occupation?',
        },
        {
            "role": "user",
            "content": "The year my pet monster come out",
        },
        {
            "role": "assistant",
            "content": 'When did "My Pet Monster" come out?',
        },
    ],
}


# TODO: few-shot learning
verify_prompt = [
    {
        "role": "system",
        # "content": (
        #     "Your task is to verify whether any of the following contexts "
        #     "contains enough information to understand the following topic. "
        #     "Please only reply 'yes' or 'no' and do not output any other words."
        # ),
        "content": (
            "Your task is to verify whether any of the following contexts "
            "contains enough information to answer the following question. "
            "Please only reply 'yes' or 'no' and do not output any other words."
        ),
    }
]


refine_prompt = {
    "bm25": {
        "extend": [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant tasked with helping users refine their search queries. "
                    "When given a context retrieved from an initial query, "
                    "your goal is to suggest additional keywords or phrases that could help obtain more relevant and specific information.\n\n"
                    "Follow these guidelines:\n"
                    "1. Analyze the Given Context: "
                    "Carefully read the provided context to understand what information has already been retrieved.\n"
                    "2. Identify Missing Information: "
                    "Determine what specific details or aspects might be missing or could be more relevant to the user's needs.\n"
                    "3. Suggest Additional Keywords: "
                    "Provide additional keywords or phrases that could help refine the search and yield more precise results. "
                    "Ensure these keywords are directly related to the topic and can help narrow down the search to more relevant information."
                ),
            },
            {
                "role": "user",
                "content": (
                    "The above context was retrieved using the given query. "
                    "However, the information may not fully address user needs. "
                    "To refine the search and obtain more relevant results, "
                    "please suggest **one** additional keyword or phrases that could help narrow down the search."
                    "Please only reply your keyword and do not output any other words."
                ),
            },
        ],
        # "filter": [
        #     {
        #         "role": "system",
        #         "content": (
        #             "The following context is retrieved using the given query. "
        #             "However, the context may not contain the information you are looking for. "
        #             "Please provide one keyword to filter out irrelevant information. "
        #         ),
        #     },
        # ],
        "emphasize": [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant tasked with helping users refine their search queries. "
                    "When given a context retrieved from an initial query, "
                    "your goal is to suggest the keywords that should be emphasized to help obtain more relevant and specific information.\n\n"
                    "Follow these guidelines:\n"
                    "1. Analyze the Given Context: "
                    "Carefully read the provided context to understand what information has already been retrieved.\n"
                    "2. Identify Missing Information: "
                    "Determine what specific details or aspects might be missing or could be more relevant to the user's needs.\n"
                    "3. Indicate Important Keywords: "
                    "Provide keywords or phrases that should be emphasized to refine the search and yield more precise results. "
                ),
            },
            {
                "role": "user",
                "content": (
                    "The above context was retrieved using the given query. "
                    "However, the information may not fully address user needs. "
                    "To refine the search and obtain more relevant results, "
                    "please suggest the keyword or phrases that should be emphasized to narrow down the search."
                    "Please only reply your keyword and do not output any other words."
                ),
            },
        ],
    },
    "dense": {},
    "web": {},
}


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
generate_prompt = {
    "with_contexts": [
        {
            "role": "system",
            "content": (
                "Answer the question based on the given contexts. "
                "Note that the context might not always contain relevant information to answer the question. "
                "Only give me the answer and do not output any other words."
            ),
        }
    ],
    "without_contexts": [
        {
            "role": "system",
            "content": (
                "Answer the following question. "
                "Only give me the answer and do not output any other words."
            ),
        }
    ],
}
