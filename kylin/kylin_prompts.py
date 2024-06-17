determine_info_prompt = (
    "Please indicate the external knowledge needed to answer the following question. "
    "If the question can be answered without external knowledge, "
    'answer "No additional information is required".\n\n'
    "For example:\n"
    "Question: who got the first nobel prize in physics.\n"
    "Needed Information: [1] The first physics Nobel Laureate.\n\n"
    "Question: big little lies season 2 how many episodes\n"
    "Needed Information: [1] The total number of episodes in the first season.\n\n"
    "Question: who sings i can't stop this feeling anymore?\n"
    'Needed Information: [1] The singer of "i can\'t stop this feeling anymore".\n\n'
    "Question: what is 1 radian in terms of pi?\n"
    "No additional information is required.\n\n"
    "Question: Which magazine was started first Arthur's Magazine or First for Women?\n"
    "Needed Information: [1] The time when Arthur's Magazine was founded.\n"
    "[2] The time when First for Women was founded.\n\n"
)


rewrite_prompt = (
    "Please optimize the following query for the {engine_name} search engine.\n"
    "{engine_desc}\n"
    "Please only reply your query and do not output any other words."
)


verify_prompt = (
    "Your task is to verify whether the following context "
    "contains enough information to understand the following topic. "
    "Please only reply 'yes' or 'no' and do not output any other words."
)


summary_prompt = (
    "Please copy the sentences from the following context that are relevant to the given question. "
    "If the context does not contain any relevant information, respond with <none>. "
    "Please only reply the relevant sentences and do not output any other words."
)


generate_prompt = (
    "Answer the question based on the given contexts. "
    "Note that the context might not always contain relevant information to answer the question. "
    "Only give me the answer and do not output any other words."
)
