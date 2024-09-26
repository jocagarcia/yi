import streamlit as st 
from snowflake.snowpark.context import get_active_session
session = get_active_session() # Get the current credentials

import pandas as pd
import datetime
from uuid import uuid4

pd.set_option("max_colwidth",None)


#### ------ Default values for app set up ------ #### 

model_name = 'mistral-7b'   # default but we allow user to select one
num_chunks = 3    # num-chunks provided as context. Play with this to check how it affects your accuracy
slide_window = 3    # how many last conversations to remember. This is the slide window.
debug = 0    # Set this to 1 if you want to see what is the text created as summary and sent to get chunks
use_chat_history = 1   # 1: Use the chat history by default
show_token_credits = 0   # 0: Not showing token consumed by default
knowledge_base = 1   # 1: use knowledge base in addition to live Snowflake doc


# credits per 1M token from credits consumption table 
# https://www.snowflake.com/legal-files/CreditConsumptionTable.pdf#page=9

credit_table = {
    'mistral-7b': 0.12,
    'mixtral-8x7b': 0.22,
    'reka-flash': 0.45,
    'mistral-large': 5.1,
    'llama3.1-70b': 1.21,
    'llama3.1-405b': 5
}



#### ------ Helper functions ------ #### 

def get_relevant_context (question):

    # obtain live, real time information from Snowflake doc website

    # query rewrite when initial user query is too long
    if len(question) > 128:
        rewrite_query = session.sql(f"""SELECT snowflake.cortex.complete(
                                          'llama3-8b', 
                                          'Find the question in the query, rewrite the question so that its length is within 128 characters, and only output the question. query: {question}'
                                          ) as rewrite_qry;"""
                               ).to_pandas()['REWRITE_QRY'][0]
    else:
        rewrite_query = question
        
    # call snowdoc_search to scrape relevant documentation pages
    df_snowdoc_search = session.sql(f"""SELECT snowdoc_search('{rewrite_query}') AS output,
                                               output[0] as response,
                                               output[1] as reference_url """
                                   ).to_pandas()

    prompt_context = ""
    
    if knowledge_base == 1:
        createsql = f"""
            create or replace table query_vec(
                qvec vector(float, 768)
            )
        """
        session.sql(createsql).collect()
    
        insertsql = f"""
            insert into query_vec 
              select snowflake.cortex.embed_text_768('e5-base-v2', '{question}')
        """
    
        session.sql(insertsql).collect()

        # vector similarity search
        cmd = f"""
            with results as
            (select RELATIVE_PATH,
                    VECTOR_COSINE_SIMILARITY(docs_chunks_table.chunk_vec, query_vec.qvec) as distance,
                    chunk
             from docs_chunks_table, query_vec
             order by distance desc
             limit {num_chunks}
            )
            select chunk, relative_path from results 
        """
    
        df_chunks = session.sql(cmd).to_pandas()       
    
        df_chunks_length = len(df_chunks) -1
    
        similar_chunks = ""
        for i in range (0, df_chunks_length):
            similar_chunks += df_chunks._get_value(i, 'CHUNK')

        # concatenate knowledge base chunks and snowdoc search result
        try:
            prompt_context = similar_chunks + df_snowdoc_search['RESPONSE'][0]
        except:
            pass

        # get pdf doc relative_path
        pdf_relative_path =  df_chunks._get_value(0,'RELATIVE_PATH')
        cmd2 = f"select GET_PRESIGNED_URL(@docs, '{pdf_relative_path}', 360) as pdf_url from directory(@docs)"
        df_url_link = session.sql(cmd2).to_pandas()
        pdf_url_link = df_url_link._get_value(0,'PDF_URL')
    
    else:
        try:
            prompt_context = df_snowdoc_search['RESPONSE'][0]
            pdf_relative_path = None
            pdf_url_link = None
        except:
            pass


    prompt_context = prompt_context.replace("'", "")

    reference_url = df_snowdoc_search['REFERENCE_URL'][0]
             
    return prompt_context, reference_url, pdf_relative_path, pdf_url_link


def get_chat_history():
#Get the history from the st.session_stage.messages according to the slide window parameter
    
    chat_history = ""
    
    start_index = max(0, len(st.session_state.messages) - slide_window)
    for i in range (start_index , len(st.session_state.messages) -1):
        role = st.session_state.messages[i]["role"]
        content = st.session_state.messages[i]["content"]
        chat_history += f"[{role}] : {content} "

    return chat_history

    
def summarize_question_with_history(chat_history, question):
# To get the right context, use the LLM to first summarize the previous conversation
# This will be used to get embeddings and find similar chunks in the docs for context
# that extend the question
    
    prompt = f"""
        '
        Based on the chat history below and the question, generate a query 
        with the chat history provided. Stay high level. Do not be too specific.
        The query should be in natual language. Output query should be less than 20 words. 
        
        CHAT HISTORY: {chat_history}
        QUESTION: {question}'
        """

    prompt = prompt.replace("'", "")
    
    cmd = """SELECT SNOWFLAKE.CORTEX.COMPLETE('%s',
                                              [{'role': 'user',
                                                'content': '%s'}],
                                              {}
                        ) as output,
                    TRIM(GET(output:choices,0):messages,'" ') as response,
                    output:usage:total_tokens::int as total_tokens;
          """ % (model_name, prompt)
    
    response = session.sql(cmd).collect()
    summary = response[0].RESPONSE
    summary_tot_tokens = response[0].TOTAL_TOKENS

    if debug ==1:
        st.text("Prompt used to generate summary:")
        st.caption(prompt)
        st.text("Summary used to find similar chunks in the docs:")
        st.caption(sumary)

    summary = summary.replace("'", "")

    return summary, summary_tot_tokens


def create_prompt (question):

    if use_chat_history == 1:
        chat_history = get_chat_history()

        if chat_history != "": #There is chat_history, so not first question
            question_summary, summary_tot_tokens = summarize_question_with_history(chat_history, question)
            prompt_context, reference_url, pdf_relative_path, pdf_url_link =  get_relevant_context(question_summary)
        else:
            prompt_context, reference_url, pdf_relative_path, pdf_url_link = get_relevant_context(question) #First question when using history
            summary_tot_tokens = 0
    else:
        prompt_context, reference_url, pdf_relative_path, pdf_url_link = get_relevant_context(question)
        chat_history = ""
        summary_tot_tokens = 0
  
    prompt = f"""
          'You are an Snowflake expert chat assistance that extracts information from the CONTEXT provided.
           You offer a chat experience considering the information included in the CHAT HISTORY.
           When ansering the question be concise, structured and do not hallucinate. 
           If you don´t have the information just say so.
           
           Do not mention the CONTEXT used in your answer.
           Do not mention the CHAT HISTORY used in your asnwer.
           
          CHAT HISTORY: {chat_history}
          CONTEXT: {prompt_context}, 
          QUESTION:  
           {question} 
           Answer: '
           """

    return prompt, reference_url, pdf_relative_path, pdf_url_link, summary_tot_tokens


@st.cache_data(show_spinner = False)  #reuse result if repeat query
def complete(question):

    prompt, reference_url, pdf_relative_path, pdf_url_link, summary_tot_tokens = create_prompt(question)

    prompt = prompt.replace("'","")

    cmd = """SELECT SNOWFLAKE.CORTEX.COMPLETE('%s',
                                              [{'role': 'user',
                                                'content': '%s'}],
                                              {}
                        ) as output,
                    TRIM(GET(output:choices,0):messages,'" ') as response,
                    output:usage:total_tokens::int as total_tokens;
          """ % (model_name, prompt)
    
    df_response = session.sql(cmd).collect()
    return df_response, reference_url, pdf_relative_path, pdf_url_link, summary_tot_tokens


def reset_conversation():
    st.session_state.messages = []

def expander_state():
    st.session_state.expand = True



####  ------ MAIN CODE ------ ####

#def main():
st.title("❄️ SnowChat for Snowflake Doc")
st.caption(
    f"""Hi! 
    I'm a chatbot to help answer any product questions you may have about Snowflake.
    """)    

st.write(' ')   # create additional row space 

with st.expander("Configure options"):
    # Give user the option to select model
    model_name = st.selectbox('Select desired model:',('mistral-7b', 'mixtral-8x7b', 'reka-flash', 'mistral-large', 'llama3.1-70b', 'llama3.1-405b'), index = 0)
    
    # For educational purposes. Users can chech the difference when using memory or not
    use_chat_history = st.toggle('Remember the chat history', value = True)

    # Users can choose to use knowledge base in addition to live Snowflake doc
    knowledge_base = st.toggle('Use knowledge base as source in addition to live Snowflake doc', value = True)

    # Users can choose to show tokens and credits or not
    show_token_credits = st.toggle('Show total tokens and credits consumed per query', value = False)
    
    # User can select debug mode
    #debug = st.toggle('Debug mode: Click to see summary generated of previous conversation')

# Log current user
user_name = st.experimental_user.user_name

st.write(' ')
st.write(' ')


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    

# Initialize session id and query sequence
if len(st.session_state.messages) == 0:
    st.session_state.id = str(uuid4())

# Initialize with sample queries.
init_placeholder = st.empty()

if len(st.session_state.messages) == 0:
    with init_placeholder.container():
        st.write(' ')
        st.write(' ')

        st.caption('Try searching:')
        st.text(' - What is unique about Snowflake architecture?')
        st.text(' - Is hybrid table available in Azure?')
        st.text(' - List out all Cortex LLM functions and available models')
        st.text(' - Compare dynamic table with streams and tasks')
        st.text(' - Give me an example of how to use the Cortex Anomaly Detection function')


else:
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        
        if message["role"] == 'user':
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        else:
            with st.chat_message(message["role"], avatar = "❄️"):
                st.markdown(message["content"])
                st.write(' ')
                st.markdown(message["reference"])
                st.write(' ')
                st.markdown(":snowboarder: Check out relevant product [demos](%s) (right click and open in a new tab)." % message["demo"])
    
                # show tokens and credits
                if show_token_credits:
                    st.write(' ')
                    st.caption(message["token"])
    
                if message["suggestion"]:
                    suggestion = message["suggestion"]
                else:
                    suggestion = "Suggest an alternative answer"
    
    
            with st.expander("Provide feedback", expanded=st.session_state.expand):
                
                st.caption("Do you like this response? Feel free to suggest an alternative answer either way.")
                
                with st.container():
                    col1, _, col3 = st.columns([4, 0.5, 12])
                
                    with col1:
                        st.write('<div style="height: 20px;"> </div>', unsafe_allow_html=True)
                        feedback = st.select_slider(
                            '',
                            options=['Yes', 'Neutral', 'No'],
                            value = message["slider"],
                            key = message["slider_key"],
                            on_change = expander_state
                        )
                        
                    with col3:
                        suggest_input = st.text_area("", placeholder=suggestion, height=80, key = message["text_key"])
    
                with st.container():
                    col1, col2 = st.columns([7, 1])                   
                    with col2:
                        # Submit button to overwrite feedback and suggestion
                        if st.button('Submit', key = message["submit_key"]):
                            suggest_input = suggest_input.replace("'", "")
                            insert_cmd = f"""
                                   UPDATE CHAT_HISTORY_LOG
                                     SET FEEDBACK = '{feedback}', SUGGESTION = '{suggest_input}'
                                     WHERE SESSION_ID = '{message["session_id"]}'
                                     AND QUERY_SEQ = {message["query_seq"]}
                                   """
                            session.sql(insert_cmd).collect()
                            st.toast("✅ Feedback Submitted!")
                        


# Accept user input
if question := st.chat_input("Type your question about Snowflake"): 

    init_placeholder.empty()

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(question)
    
        
    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar = "❄️"):

        with st.spinner("I'm thinking. Please wait..."):
            message_placeholder = st.empty()

            question = question.replace("'","")

            try:
                response, reference_url, pdf_relative_path, pdf_url_link, summary_tot_tokens = complete(question)
            except:
                st.markdown(':red[Max Tokens exceeded. Please click on start over button to free up chat history!]')
                st.button('Start Over', on_click=reset_conversation, key = str(uuid4()))
                

        # show response
        res_text = response[0].RESPONSE
        message_placeholder.markdown(res_text)
        #st.markdown(res_text)
        
        tot_tokens = response[0].TOTAL_TOKENS

        tot_tokens = tot_tokens + summary_tot_tokens
        
        res_text = res_text.replace("'", "")


        # show reference url and pdf relative path
        if (reference_url is not None) and (knowledge_base == 1):
            st.write(' ')
            reference_url = reference_url.replace('"', '').split(',')
            display_url = f""":book: Sources:  
            1) {reference_url[0]}  
            2) {reference_url[1]}  
            3) [{pdf_relative_path}]({pdf_url_link})"""
        elif (reference_url is not None) and (knowledge_base == 0):
            st.write(' ')
            reference_url = reference_url.replace('"', '').split(',')
            display_url = f""":book: Sources:  
            1) {reference_url[0]}  
            2) {reference_url[1]}"""
        elif knowledge_base:
            display_url = f""":book: Sources:  
            1) [{pdf_relative_path}]({pdf_url_link})"""            
        else:
            display_url = ''

        st.markdown(display_url)

        
        # show youtube demos
        
        # rewrite query to extract key words for more effective searching
        if len(question) > 128:
            rewrite_query = session.sql(f"""SELECT snowflake.cortex.complete(
                                          'llama3-70b', 
                                          'Is there a relevant Snowflake key feature, use case or workload that is mentioned directly in the query? 
                                           If yes, return the name of the feature, use case or workload. 
                                           If not, say no. Be extremely concise and only answer the question as instructed.  
                                           Query: {question}'
                                          ) as rewrite_qry;"""
                               ).to_pandas()['REWRITE_QRY'][0]
        else:
            rewrite_query = question

        # display 
        st.write(' ')
        youtube_url = "https://www.youtube.com/@snowflakedevelopers/search?query=%s" % rewrite_query.replace('?', '').replace(':', '').replace('\n', '%20').replace(' ', '%20')
        st.markdown(":snowboarder:  Check out relevant product [demos](%s) (right click and open in a new tab)." % youtube_url)

        
        # show tokens and credits
        display_token = f"""Total tokens: {tot_tokens}. 
                            Total credits: {round(tot_tokens * credit_table[model_name] / 1000000, 6)}
                        """
        if show_token_credits:
            st.write(' ')
            st.caption(display_token)
        else:
            pass


    # Control query sequence in a session
    if len(st.session_state.messages) == 0:
        st.session_state.query_seq = 1
    else:
        st.session_state.query_seq += 1
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question, "reference": ''})

    
    
    # Response evaluation
    with st.expander("Provide feedback", expanded=False):

        st.session_state.expand = False
        
        st.caption("Do you like this response? Feel free to suggest an alternative answer either way.")
        
        with st.container():
            col1, _, col3 = st.columns([4, 0.5, 12])
        
            with col1:
                slider_key = str(uuid4())
                
                st.write('<div style="height: 20px;"> </div>', unsafe_allow_html=True)
                feedback = st.select_slider(
                    '',
                    options=['Yes', 'Neutral', 'No'],
                    value = 'Neutral',
                    key = slider_key,
                    on_change = expander_state
                )
                
            with col3:
                text_key = str(uuid4())
                suggest_input = st.text_area("", placeholder="Suggest an alternative answer", height=80, key = text_key)

            with st.container():
                col1, col2 = st.columns([7, 1])                   
                with col2:
                    submit_key = str(uuid4())
                    st.button('Submit', key = submit_key)

    
    st.session_state.messages.append({"role": "assistant",
                                      "session_id": st.session_state.id,
                                      "query_seq": st.session_state.query_seq,
                                      "content": res_text, 
                                      "reference": display_url,
                                      "demo": youtube_url,
                                      "token": display_token,
                                      "slider_key": slider_key,
                                      "text_key": text_key,
                                      "submit_key": submit_key,
                                      "slider": feedback,
                                      "suggestion": suggest_input
                                     })

    

    # Write query and response back to chat_history_log table
    ts = session.sql("SELECT CURRENT_TIMESTAMP(0) :: varchar as ts").to_pandas()['TS'][0]
    df = pd.DataFrame({"TIMESTAMP": [ts], "SESSION_ID": [st.session_state.id], "QUERY_SEQ": [st.session_state.query_seq]
                       , "USER_NAME": [user_name], "QUERY": [question], "RESPONSE": [res_text]
                       , "FEEDBACK": [feedback], "SUGGESTION": [suggest_input]})
    session.write_pandas(df, "CHAT_HISTORY_LOG")


st.write(' ')
st.write(' ')


# start over button
if st.session_state.messages:
    st.button('Start Over', on_click=reset_conversation, key = str(uuid4()))



# if __name__== '__main__':
#     main()
