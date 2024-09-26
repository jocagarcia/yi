// Create a database and schema for SnowChat
CREATE DATABASE IF NOT EXISTS snowflake_chatbot;
CREATE SCHEMA IF NOT EXISTS snowchat;

USE DATABASE snowflake_chatbot;
USE SCHEMA snowchat;


// Create a stage for storing pdf
create or replace stage docs 
ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE') 
DIRECTORY = (ENABLE = true)
;


// Download pdfs from the link below and upload them to the stage docs
-- https://drive.google.com/drive/folders/1uSVzHPJwKau-zJ5MOz-YsWpAaaq_1zId?usp=sharing


// Check docs in stage
ls @docs;


// Create UDF pdf_text_chunker
create or replace function pdf_text_chunker(file_url string, size integer, overlap integer)
returns table (chunk varchar)
language python
runtime_version = '3.9'
handler = 'pdf_text_chunker'
packages = ('snowflake-snowpark-python','PyPDF2', 'langchain')
as
$$
from snowflake.snowpark.types import StringType, IntegerType, StructField, StructType
from langchain.text_splitter import RecursiveCharacterTextSplitter
from snowflake.snowpark.files import SnowflakeFile
import PyPDF2, io
import logging
import pandas as pd

class pdf_text_chunker:

    def read_pdf(self, file_url: str) -> str:
    
        logger = logging.getLogger("udf_logger")
        logger.info(f"Opening file {file_url}")
    
        text = ""
        with SnowflakeFile.open(file_url, 'rb') as file:
            f = io.BytesIO(file.readall())
            pdf_reader = PyPDF2.PdfFileReader(f)
            text = ""
            for page in pdf_reader.pages:
                try:
                    text += page.extract_text().replace('\n', ' ').replace('\0', ' ')
                except:
                    text = "Unable to Extract"
                    logger.warn(f"Unable to extract from file {file_url}, page {page}")
        return text

    def process(self, file_url: str, size, overlap):

        text = self.read_pdf(file_url)
        
        text_splitter = RecursiveCharacterTextSplitter(
            #separators = ["\n"],
            chunk_size = size,   #Adjust this as you see fit
            chunk_overlap  = overlap,   #This let's text have some form of overlap. Useful for keeping chunks contextual
            length_function = len
            #add_start_index = True #Optional but useful if you'd like to feed the chunk before/after
        )
    
        #chunks = text_splitter.create_documents(text)
        #df = pd.DataFrame(chunks, columns=['chunks', 'meta'])

        chunks = text_splitter.split_text(text)
        df = pd.DataFrame(chunks, columns=['chunks'])
        
        yield from df.itertuples(index=False, name=None)
$$
;


// Build the Vector Store

create or replace table DOCS_CHUNKS_TABLE ( 
    RELATIVE_PATH VARCHAR(16777216), -- Relative path to the PDF file
    SIZE_MB NUMBER(38,2), -- Size of the PDF
    FILE_URL VARCHAR(16777216), -- URL for the PDF
    SCOPED_FILE_URL VARCHAR(16777216), -- Scoped url (you can choose which one to keep depending on your use case)
    CHUNK VARCHAR(16777216), -- Piece of text
    CHUNK_VEC VECTOR(FLOAT, 768)  -- Embedding using the VECTOR data type
)
;  


-- insert doc chunks
insert into docs_chunks_table (relative_path, size_mb, file_url,
                            scoped_file_url, chunk, chunk_vec)
    select relative_path, 
            round(size / 1000000, 2) as size_mb,
            file_url, 
            build_scoped_file_url(@docs, relative_path) as scoped_file_url,
            func.chunk as chunk,
            snowflake.cortex.embed_text_768('e5-base-v2',chunk) as chunk_vec
    from 
        directory(@docs),
        TABLE(pdf_text_chunker(build_scoped_file_url(@docs, relative_path), 2000, 300)) as func
;


-- Check it out
select * from docs_chunks_table limit 5;

select relative_path, count(*) as num_chunks 
from docs_chunks_table
group by relative_path
;



// Create external Access to Snowflake Website

CREATE OR REPLACE NETWORK RULE snowdoc_network_rule
  MODE = EGRESS
  TYPE = HOST_PORT
  VALUE_LIST = ('docs.snowflake.com', 'other-docs.snowflake.com');


-- You need ACCOUNTADMIN to use external access integration
USE ROLE accountadmin;

CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION snow_doc_access_integration
ALLOWED_NETWORK_RULES = (snowdoc_network_rule)
ENABLED = true;


-- Create UDF that scrapes Snowflake Documentation through Search Bar
CREATE OR REPLACE FUNCTION snowdoc_search(topic STRING)
RETURNS ARRAY
LANGUAGE PYTHON
RUNTIME_VERSION = 3.9
HANDLER = 'get_links'
EXTERNAL_ACCESS_INTEGRATIONS = (snow_doc_access_integration)
PACKAGES = ('requests', 'beautifulsoup4')
--SECRETS = ('cred' = oauth_token )
AS
$$
import _snowflake
import requests
from bs4 import BeautifulSoup

def get_links(keyword):
  url = f"https://docs.snowflake.com/search?q={keyword}"
  response = requests.get(url)
  soup = BeautifulSoup(response.text, 'html.parser')
  matches = ["-guide/", "collaboration", "sql-reference"]
  links = [link.get("href") for link in soup.find_all("a") if any(x in link.get('href') for x in matches)]

  try:
      response2 = requests.get(links[0])
      soup2 = BeautifulSoup(response2.text)
      
      response3 = requests.get(links[1])
      soup3 = BeautifulSoup(response3.text)

      result = soup2.get_text() + ' ' + soup3.get_text()
      
      return [result.replace('\n', ' '), links[0]+ ', ' +links[1]]
  except:
      return []
$$
;


-- Test function
select snowdoc_search('What is Cortex AI?') as output,
       output[0] as response,
       output[1] as reference_url      
;



// Automatic processing of new docs
    
create or replace stream docs_stream on stage docs;

create or replace task task_extract_chunk_vec_from_pdf 
    warehouse = SMALL_WH
    schedule = '180 minute'
    when system$stream_has_data('docs_stream')
    as

    insert into docs_chunks_table (relative_path, size_mb, file_url,
                                scoped_file_url, chunk, chunk_vec)
        select relative_path, 
                round(size / 1000000, 2) as size_mb,
                file_url, 
                build_scoped_file_url(@docs, relative_path) as scoped_file_url,
                func.chunk as chunk,
                snowflake.cortex.embed_text_768('e5-base-v2',chunk) as chunk_vec
        from 
            docs_stream,
            TABLE(pdf_text_chunker(build_scoped_file_url(@docs, relative_path), 3000, 300)) as func;
        

alter task task_extract_chunk_vec_from_pdf resume;

alter task task_extract_chunk_vec_from_pdf suspend;


-- manually execute task if needed
select system$stream_has_data('docs_stream');

EXECUTE TASK task_extract_chunk_vec_from_pdf;




// Table to save logs (timestamp, session, user, query, response, feedback)
    
create or replace table chat_history_log ( 
    TIMESTAMP VARCHAR,
    SESSION_ID VARCHAR,
    QUERY_SEQ INT,
    USER_NAME STRING,
    QUERY VARCHAR, 
    RESPONSE VARCHAR,
    FEEDBACK VARCHAR,
    SUGGESTION VARCHAR
)
; 

select * 
from chat_history_log 
order by timestamp desc
;




