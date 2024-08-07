import os 
from dotenv import load_dotenv
from functools import wraps
from flask import Flask, jsonify, Response, request
import flask
from cache import MemoryCache
from langchain_community.utilities.sql_database import SQLDatabase
from vanna.openai import OpenAI_Chat
from openai import AzureOpenAI
from vanna.chromadb import ChromaDB_VectorStore
from typing_extensions import Annotated
import random
import string
import json
from datetime import date
import pyodbc
import autogen
import ast
from sqlalchemy.engine import URL
import openai
from langchain_community.llms import OpenAI
from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain.chains import ConversationChain


from sqlalchemy.engine import URL
from sqlalchemy import create_engine
import autogen
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
import os
# import autogen, random, string, json, ast
# from typing_extensions import Annotated
import pyodbc
import json
import ast
# from datetime import date
from sqlalchemy.engine import URL
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine   
load_dotenv()
app = Flask(__name__, static_url_path='')

# SETUP
cache = MemoryCache()

os.environ["AZURE_OPENAI_API_KEY"] = 'e89e3355197a47ff825c7d0e78582e0c'
os.environ["AZURE_OPENAI_ENDPOINT"] = 'https://api.geneai.thermofisher.com/dev/gpt4o'
os.environ["AZURE_OPENAI_API_VERSION"] = '2024-05-01-preview'
os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "gpt-4o"




class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)

        azure_openai_client = AzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        )
        OpenAI_Chat.__init__(self, client=azure_openai_client, config=config)


vn = MyVanna(config={'model': 'GPT-4o'})




# NO NEED TO CHANGE ANYTHING BELOW THIS LINE
def requires_cache(fields):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            id = request.args.get('id')


            if id is None:
                return jsonify({"type": "error", "error": "No id provided"})

            for field in fields:
                if cache.get(id=id, field=field) is None:
                    return jsonify({"type": "error", "error": f"No {field} found"})

            field_values = {field: cache.get(id=id, field=field) for field in fields}

            # Add the id to the field_values
            field_values['id'] = id

            return f(*args, **field_values, **kwargs)

        return decorated

    return decorator


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})


@app.route('/api/v0/db/connect', methods=['POST'])
def connect_to_db():
    server = flask.request.json.get('server')
    database = flask.request.json.get('database')
    username = flask.request.json.get('username')
    password = flask.request.json.get('password')

    try:
        vn.connect_to_mssql(
            odbc_conn_str='DRIVER={ODBC Driver 18 for SQL Server};TrustServerCertificate=yes;SERVER=' + server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password + ';timeout=30')
        vn.run_sql("select max(table_catalog) as x from information_schema.tables")
        return jsonify({"status": "connected"})
    except Exception as e:
        return flask.make_response(jsonify({"type": "error", "error": str(e)}), 500)




@app.route('/api/v0/table_details', methods=['GET'])
def table_details():
    result = description_agent()
    return jsonify(result)




import os
import autogen, random, string, json, ast
from typing_extensions import Annotated
import pyodbc
import json
import ast
from datetime import date
from sqlalchemy.engine import URL
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
os.environ["AZURE_OPENAI_API_KEY"] = 'e89e3355197a47ff825c7d0e78582e0c'
os.environ["AZURE_OPENAI_ENDPOINT"] = 'https://api.geneai.thermofisher.com/dev/gpt4o'
os.environ["AZURE_OPENAI_API_VERSION"] = '2024-05-01-preview'
os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "gpt-4o"


current_table_number = -1
total_tables_in_database = 2

new_json_datas = []

COLUMN_BATCH_SIZE = 20
last_table_name = ''
last_table_batches = ''
last_table_batch_number = 0

global_max_round_of_interaction = 100
N = 128

random_termination_key = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(N))



def description_agent():
    import os

    # current_table_number = -1
    # total_tables_in_database = 4

    # new_json_datas = []

    # COLUMN_BATCH_SIZE = 20
    # last_table_name = ''
    # last_table_batches = ''
    # last_table_batch_number = 0

    # global_max_round_of_interaction = 100
    # N = 128

    # random_termination_key = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(N))

    task = '''
    A user (here on referred to as user as well as Admin) who knows nothing about LLMs,
    or SQL or anything technical, wants to check whether LLMs can understand and connect
    to Databases. The tables that are to be used in this experiment are 'VW_DPE_SNOW_RFCs',
    'VW_DPE_SNOW_Incidents'.
    DO NOT USE ANY OTHER TABLES. DO NOT TRY TO ACCESS THE DATABASE IN ANY MANNER.
    ALL THE REQUIRED INFORMATION IS ALREADY PROVIDED.
    USE DOUBLE QUOTES FOR WITHIN JSON DEFINITIONS. DO NOT USE SINGLE QUOTES.

    The order of what to do is as follows:

    1. First, get the table details for all tables like table name, column name,
        Data Type and sample Unique values in JSON format. This will be provided by the
        Admin/User. There are 4 tables in the Database. The Admin will give the table details
        one by one, when the function get_json() is called. DO NOT CALL get_json() FUNCTION
        MORE THAN ONCE IN A SINGLE SITTING. The Admin will provide the next table details
        only after the other steps have been executed.

    2. For each of these values pairs, add the 'Description' of the data as an extra piece of information.
        The description should be in human-readable format, input in string format. For example,
        if the column name is 'Colours' and the data is 'Red, Blue, Green', the description should be
        'This column contains the colours with the values being Red, Blue, and Green'. DO NOT ASSUME ANYTHING
        REGARDING THE DATA. IF SOME TABLE NAME/COLUMN NAME/VALUES WITHIN ARE NOT CLEAR, WRITE
        "MORE CONTEXT REQUIRED". Fill it for the every column for every table, there is no need to ask
        for confirmation for each column. Just fill it in. IF ANY ERROR ARISES, RETRY WITH THE WHOLE
        INFORMATION, NOT A SMALLER PART EVER. WHEN PASSING THE UPDATED SCHEMA BACK USING
        THE make_json() FUNCTION, PASS IT IN THE FORMAT OF:
        
            Table Name: VW_DPE_SNOW_Incidents
            Data Type: varchar
            Column Name: number
            Unique Values: INC6102881, INC6103117, INC6103932, INC6106402, INC6106962, INC6107980, INC6108852, INC6108959, INC6109607, INC6109936
            Description: The column is from the table

    3. Return the resulting JSON file as a string to the Admin/User using the built-in function make_json(),
        by passing it to the function as a string. Do this once for each table, just after adding the description.
        This will be done for each table, one by one. The Admin will provide the next table details
        until all the 4 tables are done.

    IMPORTANT: DO NOT ASSUME ALL THE TABLES HAVE BEEN PROCESSED. KEEP CALLING get_json() FUNCTION
    UNTIL A TERMINATION STRING IS GIVEN BY THE FUNCTION. DO NOT STOP UNTIL THE TERMINATION STRING IS GIVEN.
    '''


    llm_config = {"model": os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
                "azure_endpoint":"https://api.geneai.thermofisher.com/dev/gpt4o",
                "api_key":os.environ.get("f7f2e7f038b64713a51760ac1c08b53d"),
                "api_type":'azure',
                "api_version":os.environ.get("AZURE_OPENAI_API_VERSION"),
                "timeout":30 
    }

    user_proxy = autogen.ConversableAgent(
        name="Admin-or-User-Proxy",
        system_message="Give the task, and send "\
        "instructions to Analyst/Planner to refine their outputs.",
        code_execution_config=False,
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: "{}".format(random_termination_key) in msg["content"],
    )
    
    planner = autogen.ConversableAgent(
        name="Planner",
        system_message="USE CHAIN OF THOUGHT TO EXPLAIN ANYTHING AND EVERYTHING."\
        "Given a task, please determine "\
        "what information and procedure is needed to complete the task. "\
        "After every step is done by others, check the progress and "\
        "instruct the remaining steps. If a step fails, try to "\
        "workaround. Do not provide the code or SQL query to be run,"\
        " as this is the Engineers' task. Merely give the steps"\
        " on what the Engineer has to do. DO NOT PROVIDE CODE."\
        "MERELY PROVIDE INSTRUCTIONS."\
        " IMPORTANT: DO NOT ASSUME ALL THE TABLES HAVE BEEN PROCESSED. KEEP CALLING get_json() FUNCTION"\
        " UNTIL A TERMINATION STRING IS GIVEN BY THE FUNCTION. DO NOT STOP UNTIL THE TERMINATION STRING IS GIVEN.",
        human_input_mode="NEVER",
        description="Planner. Given a task, determine what "
        "information is needed to complete the task. "
        "After each step is done by others, check the progress and "
        "instruct the remaining steps",
        llm_config=llm_config,
        is_termination_msg=lambda msg: "{}".format(random_termination_key) in msg["content"],
    )

    analyst = autogen.ConversableAgent(
        name="Analyst",
        llm_config=llm_config,
        system_message="Analyst."
        # "USE CHAIN OF THOUGHT TO EXPLAIN ANYTHING AND EVERYTHING."\
        # "Please write the content in human-readable format (with relevant titles)"\
        # " and put the content in pseudo ```md``` code block if required. "\
        # "You take feedback from the admin and refine your result."
        "Do not provide the code or SQL query to be run,"\
        " as this is the Engineers' task. Merely analyse the data"\
        " which has been given. DO NOT PROVIDE CODE."\
        # "MERELY PROVIDE YOUR ANALYSIS.",
        " MERELY PROVIDE WHAT HAS BEEN ASKED FOR."\
        " WHEN GIVING DESCRIPTIONS TO THE NEXT FUNCTION CALL,"\
        " IF THERE IS A NEED FOR ANY FUNCTION CALL,"\
        " JUST SUGGEST THE FUNCTION CALL TO THE NEXT AGENT WITHOUT ANY"\
        " EXPLANATION. IF THE NEXT STEP IS TO BE A FUNCTION CALL,"\
        " JUST SUGGEST THE FUNCTION CALL AND ITS ARGUMENTS TO THE NEXT AGENT WITHOUT ANY"\
        " EXPLANATION. DO NOT PROVIDE DESCRIPTIONS AND ATTRIBUTES IN HUMAN READABLE FORMAT."\
        " IMPORTANT: DO NOT ASSUME ALL THE TABLES HAVE BEEN PROCESSED. KEEP CALLING get_json() FUNCTION"\
        " UNTIL A TERMINATION STRING IS GIVEN BY THE FUNCTION. DO NOT STOP UNTIL THE TERMINATION STRING IS GIVEN.",
        human_input_mode="NEVER",
        description="Analyst."
        "Write content based on the code execution results and take "
        "feedback from the admin to refine it further.",
        is_termination_msg=lambda msg: "{}".format(random_termination_key) in msg["content"],
    )
    
    def getDatabaseInfo(table_number):

        server = 'Awsu1-10wpcd02p.amer.thermo.com'
        database = 'data_analytics'
        username = 'reportsalmpcexternal'
        password = 'Almpc@3@Thermo'
        driver= '{ODBC Driver 18 for SQL Server}'
        
        connection_string = f'DRIVER={driver};TrustServerCertificate=yes;SERVER={server};DATABASE={database};UID={username};PWD={password}'
        
        connection_url  = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})
        
        engine = create_engine(connection_url)
        # db = SQLDatabase(engine,
        #                 include_tables=['VW_DPE_SNOW_RFCs',
        #                                 'VW_DPE_SNOW_Incidents',
        #                                 'VW_DPE_SNOW_RFCs_INCs_ChangeFailres',
        #                                 'dpe_masterdashboard_fact_teams'],
        #                 sample_rows_in_table_info=2,
        #                 view_support=True)

        db = SQLDatabase(engine,
                include_tables=['VW_DPE_SNOW_RFCs',
                                'VW_DPE_SNOW_Incidents',
                                ],
                sample_rows_in_table_info=2,
                view_support=True)

        # table_list = ['VW_DPE_SNOW_RFCs',
        #               'VW_DPE_SNOW_Incidents',
        #               'VW_DPE_SNOW_RFCs_INCs_ChangeFailres',
        #               'dpe_masterdashboard_fact_teams']

        table_list = ['VW_DPE_SNOW_RFCs',
                'VW_DPE_SNOW_Incidents']

        class DateEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, date):
                    return obj.isoformat()
                return json.JSONEncoder.default(self, obj)

        def fetch_table_and_column_names():
            table_names = [table_list[table_number]] #['VW_DPE_SNOW_Incidents']
            table_names_str = ', '.join(f"'{table_name}'" for table_name in table_names)
            query = f"""
            SELECT TABLE_NAME, COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME IN ({table_names_str})
            ORDER BY TABLE_NAME, ORDINAL_POSITION
            """
            result = db.run(query)
            if isinstance(result, str):
                result = ast.literal_eval(result)
            tables = {}
            for row in result:
                if row[0] not in tables:
                    tables[row[0]] = []
                tables[row[0]].append(row[1])
            return tables

        def fetch_column_data_and_unique_values(table_name, columns):
            results = []
            with pyodbc.connect(connection_string) as conn:
                cursor = conn.cursor()
                for column in columns:
                    query_data_type = f"""
                    SELECT DATA_TYPE
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_NAME = '{table_name}' AND COLUMN_NAME = '{column}'
                    """
                    cursor.execute(query_data_type)
                    data_type = cursor.fetchone()[0]
                    query_unique_values = f"""
                    SELECT DISTINCT [{column}]
                    FROM {table_name}
                    WHERE [{column}] IS NOT NULL
                    ORDER BY [{column}]
                    OFFSET 0 ROWS FETCH NEXT 3 ROWS ONLY
                    """
                    cursor.execute(query_unique_values)
                    unique_values = [row[0] for row in cursor.fetchall()]
                    results.append({
                        "column_name": column,
                        "data_type": data_type,
                        "unique_values": unique_values,
                        "description":"" # Add description here
                    })
            return results

        tables = fetch_table_and_column_names()
        data = []
        for table_name, columns in tables.items():
            table_data = {
                "table_name": table_name,
                "columns": fetch_column_data_and_unique_values(table_name, columns)
            }
            data.append(table_data)

        json_data = json.dumps({"tables": data}, indent=4, cls=DateEncoder)
        

        return json_data

    @user_proxy.register_for_execution()
    @planner.register_for_llm(description="Get JSON file containing the table details.")
    @analyst.register_for_llm(description="Get JSON file containing the table details.")
    #def get_json(table_name_input: Annotated[str, "The input to function get_json(). Pass the table name as the argument."]
    #) -> str:
    def get_json() -> str:
        
        global last_table_batches, last_table_name, last_table_batch_number, current_table_number
        #table_name = table_name_input

        # if table_name == last_table_name: # continuation of the previous table
        if last_table_batch_number + 1 < len(last_table_batches): #and table_name == last_table_name:
            last_table_batch_number += 1
            current_batch = str(last_table_batches[last_table_batch_number])
            if last_table_batch_number + 1 < len(last_table_batches):
                current_batch += '''
                This is the continuation of the before table. The rest of the table 
                is still too large to be displayed in a single message. After 
                calling make_json() for this data,
                call this function again with the same 
                table name as function argument for the rest of the table.
                DO NOT ASSUME THE TABLE IS OVER UNTIL IT IS EXPLICITLY MENTIONED.
                WHEN CALLING make_json() FUNCTION, ONLY PASS IN THE TABLE NAME
                AS THE ARGUMENT. DO NOT WRITE IN ANYTHING ELSE.'''
            else:
                if current_table_number == total_tables_in_database - 1:
                    current_table_number = -2
                current_batch += '''
                This table has been completely displayed.'''

        else: #new table
            current_table_number += 1
            json_file = getDatabaseInfo(current_table_number)
            current_table_batches = getJSONinBatches(json_file)
            last_table_batches = current_table_batches
            #last_table_name = table_name
            last_table_batch_number = 0
            current_batch = str(current_table_batches[0])
            if len(current_table_batches) > 1:
                current_batch += '''
                This table is too large to be displayed 
                in a single message. After calling make_json() for this data,
                call this function again with the same 
                table name as function argument for the rest of the table.
                DO NOT ASSUME THE TABLE IS OVER UNTIL IT IS EXPLICITLY MENTIONED.
                WHEN CALLING make_json() FUNCTION, ONLY PASS IN THE TABLE NAME
                AS THE ARGUMENT. DO NOT WRITE IN ANYTHING ELSE.'''
            else:
                if current_table_number == total_tables_in_database - 1:
                    current_table_number = -2
                current_batch += '''
                This table has been completely displayed.'''

        return current_batch

    def getJSONinBatches(info):
        json_data = ast.literal_eval(info)
        tables = json_data['tables']
        split_tables = []
        
        for table in tables:
            columns = table['columns']
            temp_columns = []
            for i, column in enumerate(columns, 1):
                temp_columns.append(column)
                if i % COLUMN_BATCH_SIZE == 0 or i == len(columns):
                    batch = {
                        "tables": [
                            {
                                "table_name": table["table_name"],
                                "columns": temp_columns.copy()
                            }
                        ]
                    }
                    split_tables.append(batch)
                    temp_columns.clear()
        
        return split_tables

    @user_proxy.register_for_execution()
    @planner.register_for_llm(description="Save the updated JSON file.")
    @analyst.register_for_llm(description="Save the updated JSON file.")
    def make_json(input_text_to_convert_to_json: Annotated[str, "The input JSON for conversion and saving as JSON."]
    ) -> str: 
        plaintext = input_text_to_convert_to_json.replace("\n", '')
        parsed = json.loads(plaintext)
        
        FINAL_JSON = parsed
        result = FINAL_JSON
        new_json_datas.append(result)
        output = 'JSON file saved.'
        # if total_tables_in_database == current_table_number + 1:
        if current_table_number == -2: #when every table is done
            output += ' TERMINATION KEY: ' + random_termination_key
        return output

    groupchat = autogen.GroupChat(
        agents=[user_proxy, analyst, planner],
        messages=[],
        max_round=global_max_round_of_interaction,
        allowed_or_disallowed_speaker_transitions={
            user_proxy: [analyst, planner],
            planner: [user_proxy, analyst],
            analyst: [user_proxy, planner],
        },
        speaker_transitions_type="allowed",
    )

    # Define the manager
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    groupchat_result = user_proxy.initiate_chat(manager,
                                                message=task,)
    
    return new_json_datas



# schema
import os

from langchain_openai import AzureChatOpenAI


@app.route('/api/v0/generate_sample_queries',methods=['GET'])
def ddl_curation():
        server = 'Awsu1-10wpcd02p.amer.thermo.com'
        database = 'data_analytics'
        username = 'reportsalmpcexternal'
        password = 'Almpc@3@Thermo'
        driver= '{ODBC Driver 18 for SQL Server}'

        connection_string = f'DRIVER={driver};TrustServerCertificate=yes;SERVER={server};DATABASE={database};UID={username};PWD={password}'

        connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})

        engine = create_engine(connection_url)
        db = SQLDatabase(engine,
                        include_tables=['VW_DPE_SNOW_RFCs',
                                        'VW_DPE_SNOW_Incidents',
                                        'VW_DPE_SNOW_RFCs_INCs_ChangeFailres',
                                        'dpe_masterdashboard_fact_teams'],
                        sample_rows_in_table_info=2,
                        view_support=True)
        schema = db.get_context()
        return schema
    
  


@app.route('/api/v0/generate_sample_queries',methods=['POST'])
def generate():
    server = 'Awsu1-10wpcd02p.amer.thermo.com'
    database = 'data_analytics'
    username = 'reportsalmpcexternal'
    password = 'Almpc@3@Thermo'
    driver= '{ODBC Driver 18 for SQL Server}'

    connection_string = f'DRIVER={driver};TrustServerCertificate=yes;SERVER={server};DATABASE={database};UID={username};PWD={password}'

    connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})

    engine = create_engine(connection_url)
    db = SQLDatabase(engine,
                    include_tables=['VW_DPE_SNOW_RFCs',
                                    'VW_DPE_SNOW_Incidents',
                                    'VW_DPE_SNOW_RFCs_INCs_ChangeFailres',
                                    'dpe_masterdashboard_fact_teams'],
                    sample_rows_in_table_info=2,
                    view_support=True)
    schema = db.get_context()

    description =  description_agent()
    response = generate_nl_questions_and_sql(schema, description)
    return response










def generate_nl_questions_and_sql(schema, description):
    # Configure Azure OpenAI Service API
    os.environ["OPENAI_API_VERSION"] = "2024-05-01-preview"
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://api.geneai.thermofisher.com/dev/gpt4o"
    os.environ["AZURE_OPENAI_API_KEY"] = "e89e3355197a47ff825c7d0e78582e0c"

    # Initialize OpenAI API client
    openai.api_type = "azure"
    openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
    openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")

    # Define the model deployment
    deployment_name = "gpt-4o"
    temperature = 0
    top_p = 0.1

    task = ''' Given the schema of all the tables in a MS SQL database and the description about each column in each table of the database. 
    Based on the table schema and description of the columns below, generate five sample natural language questions and their equivalent SQL queries in the below mentioned sample output format.
    Schema: {schema}
    Columns_Description :{description}
    ** Note: Sample Sample_Output_Format contains the content which is not related to our database. Just take it as reference ,not to write queries **
    Sample_Output_Format:
    [
        {{
          "question":"What are the names of all the Cities in Canada",
          "answer":"SELECT geo_name, id FROM data_commons_public_data.cybersyn.geo_index WHERE iso_name ilike '%can%'"
        }},
        {{
          "question":"what are the number of public holidays for US year on year ? ",
          "answer":"SELECT date_part('year', date) as year,count(*) as num_public_holidays FROM data_commons_public_data.cybersyn.public_holidays WHERE  geo ilike '%United States%' GROUP BY year ORDER BY year asc``` I replaced `geo_name` with `geo` in the `WHERE` clause as the error message suggests that `geo_name` is an invalid identifier."
        }},
    ]
    NOTE : STRICTLY GIVE THE JSON DIRECTLY AS OUTPUT, NO EXTRA STRINGS OR DELIMETERS.
    '''
    prompt = task.format(schema=schema, description=description)
    return chat_with_model(prompt)




# Define a function to chat with the model using LangChain
def chat_with_model(prompt):
    try:
        
        
        llm = AzureChatOpenAI(deployment_name = "gpt-4o", temperature = 0.1)
        
        
        # Run the chain with the user's prompt
        response = llm.invoke(prompt)
        response = response.content
        print("THIS IS THE RESPOSNE",response)

        # Clean the response to remove Markdown-style code fences and any extraneous text
        response = response.strip()
        if response.startswith("Sure, here are five sample natural language questions and their equivalent SQL queries based on the provided schema and column descriptions"):
            response = response[len("```json"):].strip()
        if response.endswith("```"):
            response = response[:-len("```")].strip()
        
        # Convert the response to JSON format
        try:

            json_response = json.loads(response)

        except json.JSONDecodeError:
            json_response = {"error": "Failed to decode JSON response", "response": response}

        return json_response
        # return response
        
    except Exception as e:
        return f"An error occurred: {str(e)}"

    
    
    





@app.route('/api/v0/generate_questions', methods=['GET'])
def generate_questions():
    return jsonify({
        "type": "question_list",
        "questions": vn.generate_questions(),
        "header": "Here are some questions you can ask:"
    })


@app.route('/api/v0/generate_sql', methods=['GET'])
def generate_sql():
    question = flask.request.args.get('question')

    if question is None:
        return jsonify({"type": "error", "error": "No question provided"})

    id = cache.generate_id(question=question)
    sql = vn.generate_sql(question=question)

    cache.set(id=id, field='question', value=question)
    cache.set(id=id, field='sql', value=sql)

    return jsonify(
        {
            "type": "sql",
            "id": id,
            "text": sql,
        })


# @app.route('/api/v0/run_sql', methods=['GET'])
# @requires_cache(['sql'])
# def run_sql(id: str, sql: str):
#     try:
#         df = vn.run_sql(sql=sql)

#         cache.set(id=id, field='df', value=df)

#         return jsonify(
#             {
#                 "type": "df", 
#                 "id": id,
#                 "df": df.head(10).to_json(orient='records'),
#             })

#     except Exception as e:
#         return jsonify({"type": "error", "error": str(e)})

@app.route('/api/v0/run_sql', methods=['POST'])
def run_sql():
    try : 
        #hardcode db conn : 
        connect_to_db()
        #id = flask.request.json.get('id')
        sql = flask.request.json.get('sql')

        df = vn.run_sql(sql=sql)
        return jsonify(
                {
                    "type": "df",
                    "df": df.head(10).to_json(orient='records'),
                })
    except Exception as e:
        return jsonify({"type": "error", "error": str(e)})



@app.route('/api/v0/download_csv', methods=['GET'])
@requires_cache(['df'])
def download_csv(id: str, df):
    csv = df.to_csv()

    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition":
                     f"attachment; filename={id}.csv"})


@app.route('/api/v0/generate_plotly_figure', methods=['GET'])
@requires_cache(['df', 'question', 'sql'])
def generate_plotly_figure(id: str, df, question, sql):
    try:
        code = vn.generate_plotly_code(question=question, sql=sql,
                                       df_metadata=f"Running df.dtypes gives:\n {df.dtypes}")
        fig = vn.get_plotly_figure(plotly_code=code, df=df, dark_mode=False)
        fig_json = fig.to_json()

        cache.set(id=id, field='fig_json', value=fig_json)

        return jsonify(
            {
                "type": "plotly_figure",
                "id": id,
                "fig": fig_json,
            })
    except Exception as e:
        # Print the stack trace
        import traceback
        traceback.print_exc()

        return jsonify({"type": "error", "error": str(e)})


@app.route('/api/v0/get_training_data', methods=['GET'])
def get_training_data():
    df = vn.get_training_data()

    return jsonify(
        {
            "type": "df",
            "id": "training_data",
            "df": df.head(25).to_json(orient='records'),
        })

@app.route('/api/v1/get_training_data', methods=['GET'])
def get_training_data_v1():
    id_param = request.args.get('id')
    df = vn.get_training_data()

    row = df.loc[df['id'] == id_param]
    row_dict = row.to_dict(orient='records')[0]

    return jsonify(
        {
            "type": "df",
            "id": id_param,
            "df": row_dict,
        })


@app.route('/api/v0/remove_training_data', methods=['POST'])
def remove_training_data():
    # Get id from the JSON body
    id = flask.request.json.get('id')

    if id is None:
        return jsonify({"type": "error", "error": "No id provided"})

    if vn.remove_training_data(id=id):
        return jsonify({"success": True})
    else:
        return jsonify({"type": "error", "error": "Couldn't remove training data"})


@app.route('/api/v1/train', methods=['POST'])
def add_training_data_v1():
    try:
        vn.train(ddl="""
            CREATE TABLE [VW_DPE_SNOW_Incidents] (
            number VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Created_Month] NVARCHAR(30) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Created_Year] INTEGER NULL,
            [MonthYear] NVARCHAR(61) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Created_on] DATE NULL,
            grp_name VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            sys_id VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            priority VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            short_description VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            statename VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            opened_at DATETIME NULL,
            resolved_at DATETIME NULL,
            resolution_code VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            resolved_time INTEGER NULL,
            team_name VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            art VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            valuestream VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL
        ),
                 CREATE TABLE [VW_DPE_SNOW_RFCs] (
            number VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            short_description VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            approval VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            type VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            state VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            start_date DATETIME NULL,
            end_date DATETIME NULL,
            assignment_group VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            sys_created_by VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            sys_created_on DATETIME NULL,
            parent VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            close_code VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            reason VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            close_notes VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            major_incident VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            sys_id VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            environment VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            grp_name VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            comprehensive_type VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            approval_state VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            team_name VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            art VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            valuestream VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL
        )
        ,
            CREATE TABLE [VW_DPE_SNOW_RFCs_INCs_ChangeFailres] (
            valuestream VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            art VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            teamname VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            createddate DATE NULL,
            rfcscount INTEGER NULL,
            incscount INTEGER NULL
        )
        ,
                 CREATE TABLE dpe_masterdashboard_fact_teams (
            [ID] BIGINT NULL,
            team_name VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Location] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Team Status] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Agile Framework] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Compliance Standard] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [ISO Scope] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Distribution List] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Group/Division (Science Apps)] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Business Unit (Science Apps)] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            pillar VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Sub-category] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            agile_release_train VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [DE Pillar Lead] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [DE Pillar Lead Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Business ART Director] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Business ART Director Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Solution Train Engineer] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Solution Train Engineer Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Release Train Engineer] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Release Train Engineer Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [VP] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [VP Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Senior Director] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Senior Director email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Director] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Director Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Senior Manager] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Senior Manager Email ] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Engineering manager] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Engineering Manager Email ] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Agile Coach] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Agile Coach Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Product (s)] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Product (s)2] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Product (s)3] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Product (s)4] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Product (s)5] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Product Owner] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Product Owner Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Scrum Master] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Scrum Master Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Developer] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Developer Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Developer2] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Developer2 Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Developer3] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Developer3 Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Developer4] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Developer4 Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Developer5] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Developer5 Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Developer6] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Developer6 Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Developer7] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Developer7 Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Developer8] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Developer8 Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Developer9] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Developer9 Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Developer10] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Developer10 Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Developer11] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Developer11 Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Developer12] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Developer12 Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Developer13] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Developer13 Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Developer14] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Developer14 Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
             [Developer15] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Developer15 Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Developer16] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Developer16 Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Developer17] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Developer17 Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Architect] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Architect Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Architect2] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Architect2 Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Architect3] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Architect3 Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Solution Owner] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Solution Owner Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Solution Owner2] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Solution Owner2 Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Solution Owner3] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Solution Owner3 Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Product Manager ] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Product Manager Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Product Manager2] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Product Manager2 Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Product Manager3] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Product Manager3 Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [TEM (BRM)] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [TEM (BRM) Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [UX] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [UX Email] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [UX2] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [UX Email2] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Agile Board] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Team Page] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [QPPI metrics link] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Configuration Item ] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Configuration Item 2] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Configuration Item 3] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Assignment Support Group (xMatters group)] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [assignment_group__servicenow_group_] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Created] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Modified] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Jira Project Key] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Component] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Workflow ID] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Verify User] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Verify Date] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
            [Rapid View ID] VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL
        );

        """)

        context = '''SCHEMA REFERENCE:
        - `snow` refers to the `Service Now` platform
        - ****(VERY IMPORTANT)****  In views (VW), column names are defined in schema with "\\t\\n<column_name>" format.
            - Example `CREATE TABLE [VW_DPE_SNOW_RFCs] (\n\tnumber VARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,` this means `VW_DPE_SNOW_RFCs` is a table/view that has `number` as a column name.
        - `VW_DPE_SNOW_Incidents`: This view contains records of production issues, detailing incidents reported in the Service Now platform.
        - `VW_DPE_SNOW_RFCs`: Each record in this table against a team name depicts the *DEPLOYMENT. This view tracks Request for Changes (RFCs) within the production environment, part of the change management process in Service Now.
            -It includes details about each change request, its approval status, and implementation timeline. Each entry represents an individual incident with associated metadata such as sys_id (***unique identifier for each deployment done by the particular team***).
            MORE Description on Deployment related query:
            CTE - Common Table Expression
            Description:
            First, extract unique year-month combinations from the start_date field in the VW_DPE_SNOW_RFCs table. Then calculate the number of RFCs per team for each month where the RFC state is either 'Closed', 'Review', or 'Implement'. Generate combination of all distinct teams with all the unique year-months to ensure every team has an entry for every month, even if they have no RFCs in that month. And then data from the teams CTE is joined  with the rfcsbymonth CTE to get the count of RFCs per team per month. Finally, the deployment frequency is calculated based on the count of RFCs.
            'daily' if cnt >= 28
            'weekly' if cnt >= 4
            'bi-weekly' if cnt >= 2
            'monthly' if cnt >= 1
            'no deployments' if cnt < 1
            -**Use this VIEW for all deployment related queries.**. For the time of deployment : refer createddate column inside this view. ALWAYS SHOE DEPLOYMENT COUNT WITH TEAM NAME IN DEPLOYMENT INFORMATION.
            - Sample queries : not deploying frequently means the team has less number of deployments <PAY GREAT ATTENTION TO THE  PARAMETER ASKED IN THE QUESTIION>
        `VW_DPE_SNOW_RFCs_INCs_ChangeFailres`  : USE THIS TABLE FOR CHANGE FAILURE RELATED QUERY.


            - formula to calculate Change Failure Rate : Change Failure rate = sum(incscount) / sum(rfcscount)
            - Change failure rate is evaulated based upon each individual team or valuestream, ALWAYS unless specified.
            - No other constraints are to be applied, UNLESS SPECIFIED.
            - Also more the change failure rate, the worse it is.
            - ALWAYS GIVE THE CHANGE FAILURE RATE IN PERCENTAGE( Give decimal values upto 2 decimal points.)





        (VERY VERY IMPORTANT)
        BEFORE SQL QUERY GENERATION NOTE :
        - UNLESS SPEICIFIED USED `AVERAGE` (AVG) EVERYWHERE IN THE QUERY
        - Pay close attention to the timeline specified in the question. Include date filters as necessary.
        - If the question asks for top performers or metrics, do not limit the results to a specific number unless explicitly stated in the question.
        - Ensure that the query includes the values of the metrics that are being filtered or ranked.
        - Ensure that all column references are fully qualified with their respective table aliases to avoid ambiguous column name errors.
        - when using count in an ORDER BY statement, use COUNT (*).
        - Ensure to include a conditional check to avoid division by zero errors, such as using `NULLIF` or `CASE` statements to handle divisions.
        - STRICTLY FOLLOW : The LIMIT clause is not valid in SQL Server. Instead, SQL Server uses the TOP clause to limit the number of rows returned. THEREFORE USE `TOP` IN YOUR QUERY INSTEAD OF LIMIT
        - **** VERY IMPORTANT **** (MANDATORY) : OUTPUT SQL QUERY ONLY, as I will be directly sending it to the SQL database . Do not include any additional string. JUST THE QUERY STRING
        - VERY IMPORTANT: Emphasize the period of data that is needed.
        - VERY IMPORTANT: Use the schema provided to reference the tables in your query.
        - VERY IMPORTANT: BE PARTICULAR IN CHOOSING THE RELEVANT TABLE AGAINST THE PROMPT, AND THE COLUMNS FROM THE SELECTED TABLES. BUT NOT FROM OTHER TABLES.

        '''
        vn.train(documentation=context)

        context1 = """
        - createddate - Column contains the date on which the incident was created.
        - incscount - contains the number of incidents on particular date
        - Change failure rate is evaluated based upon each individual team or valuestream.
            1. For particular team or valuestream, it is calculated by considering only the months which has total number of incscount in a particular month is greater than zero.
            2. While calculating the total number of incscount in a particular month, consider all the entries of that month.
            3. If the above conditions are satisfied, find sum of all incscount in that month /sum of all rfcscount in that month, to get the change failure rate of the team in that particular month.
            4. To calculate the overall change failure rate for a particular team or valuestream, find the sum of incscount of all the months that has total number of incscount in a month is greater than zero/ sum of rfcscount of all those months.While calculating the total number of incscount in a particular month, consider all the entries of that month.
        """
        vn.train(documentation=context1)

        vn.train(question="Calculate the number of deployments and Deployment Frequency for all the teams  ", sql='''select
            team_name,
            FORMAT(sys_created_on ,
            'yyyyMM'),
            count(*) as number_of_deployments,
            case
                when count(*) >= 30 then 'daily'
                when count(*) >= 4 then 'weekly'
                when count(*) >= 2 then 'bi-weekly'
                when count(*) >= 1 then 'monthly'
                else '> monthly'
            end as deploymentfrequncy
        from
            data_analytics.dbo.VW_DPE_SNOW_RFCs
        group by
            team_name,
            FORMAT(sys_created_on ,
            'yyyyMM');''')

        vn.train(question='What is the change failure rate of TF ChatBot from June 17, 2023 to June 6, 2024', sql='''SELECT
            (SELECT
                SUM(TotalIncidents)
            FROM
                (SELECT
                    teamname,
                    FORMAT(createddate, 'yyyyMM') AS MonthYear,
                    SUM(rfcscount) AS TotalRFCs,
                    SUM(incscount) AS TotalIncidents
                FROM
                    VW_DPE_SNOW_RFCs_INCs_ChangeFailres
                WHERE
                    teamname = 'TF ChatBot'
                    AND createddate >= '2023-06-17'
                    AND createddate <= '2024-06-06'
                GROUP BY
                    teamname,
                    FORMAT(createddate, 'yyyyMM')
                HAVING
                    SUM(incscount) > 0) AS Subquery
            ) * 100.0 / NULLIF(
                (SELECT
                    SUM(TotalRFCs)
                FROM
                    (SELECT
                        teamname,
                        FORMAT(createddate, 'yyyyMM') AS MonthYear,
                        SUM(rfcscount) AS TotalRFCs,
                        SUM(incscount) AS TotalIncidents
                    FROM
                        VW_DPE_SNOW_RFCs_INCs_ChangeFailres
                    WHERE
                        teamname = 'TF ChatBot'
                        AND createddate >= '2023-06-17'
                        AND createddate <= '2024-06-06'
                    GROUP BY
                        teamname,
                        FORMAT(createddate, 'yyyyMM')
                    HAVING
                        SUM(incscount) > 0) AS Subquery
                ), 0
            ) AS OChangeFailureRate;''')

        return jsonify({"id": "training success"})
    except Exception as e:
        print("TRAINING ERROR", e)
        return jsonify({"type": "error", "error": str(e)})


@app.route('/api/v0/train', methods=['POST'])
def add_training_data():
    question = flask.request.json.get('question')
    sql = flask.request.json.get('sql')
    ddl = flask.request.json.get('ddl')
    documentation = flask.request.json.get('documentation')

    try:
        id = vn.train(question=question, sql=sql, ddl=ddl, documentation=documentation)

        return jsonify({"id": id})
    except Exception as e:
        print("TRAINING ERROR", e)
        return jsonify({"type": "error", "error": str(e)})


@app.route('/api/v0/generate_followup_questions', methods=['GET'])
@requires_cache(['df', 'question', 'sql'])
def generate_followup_questions(id: str, df, question, sql):
    followup_questions = vn.generate_followup_questions(question=question, sql=sql, df=df)

    cache.set(id=id, field='followup_questions', value=followup_questions)

    return jsonify(
        {
            "type": "question_list",
            "id": id,
            "questions": followup_questions,
            "header": "Here are some followup questions you can ask:"
        })


@app.route('/api/v0/load_question', methods=['GET'])
@requires_cache(['question', 'sql', 'df', 'fig_json', 'followup_questions'])
def load_question(id: str, question, sql, df, fig_json, followup_questions):
    try:
        return jsonify(
            {
                "type": "question_cache",
                "id": id,
                "question": question,
                "sql": sql,
                "df": df.head(10).to_json(orient='records'),
                "fig": fig_json,
                "followup_questions": followup_questions,
            })

    except Exception as e:
        return jsonify({"type": "error", "error": str(e)})


@app.route('/api/v0/get_question_history', methods=['GET'])
def get_question_history():
    return jsonify({"type": "question_history", "questions": cache.get_all(field_list=['question'])})

###### Chat Agent
# Initialize the Azure OpenAI model
llm = AzureChatOpenAI(
    openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    temperature=0.4,
)

# Create memory with a custom token limit
memory = ConversationSummaryBufferMemory(llm=llm)

# Conversation template
template = """
The following is a curious conversation between a Database Admin and an AI context collector. The AI context collector is very curious and intelligent.
It tries to understand information about the database from the Database Admin and asks questions about metrics used in the database, how to calculate them, and SQL queries to retrieve those metrics.

Current Conversation:
{history}
Database Admin: 
{input}
AI Assistant:
"""

PROMPT = PromptTemplate(template=template, input_variables=['history', 'input'])

# Create the conversation chain
conversation = ConversationChain(
    prompt=PROMPT,
    llm=llm,
    memory=memory,
    verbose=True
)

# Detailed summary template
summary_template = """
Summarize the conversation by highlighting the main points discussed, any SQL queries suggested, key metrics mentioned, and any questions thaNOTE: YOUR TASK IS TO SUMMARISE IN SUCH A WAY THAT, THE SUMMARY CAN BE DIRECTLY FED TO VECTOR DATABASE AS A CONTEXT FOR RAG OPERATION ON THE SAME DATABASE.
VERY IMPORTANT : Take content from conversation only, not any additional information.
{input}
Conversation:
{history}
"""

summary_prompt = PromptTemplate(template=summary_template, input_variables=['history', 'input'])

# Generate a detailed summary
summary_chain = ConversationChain(
    prompt=summary_prompt,
    llm=llm,
    memory=memory,
    verbose=True
)

# Route for handling conversation
@app.route('/api/v0/chat', methods=['POST'])
def chat():
    """
    Endpoint to handle chat interactions.
    Expects JSON input with a 'message' field.
    Returns JSON with 'response' field containing the AI's reply.
    """
    data = request.json
    user_input = data.get('message')
    
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    
    response = conversation.predict(input=user_input)
    memory.save_context({"input": user_input}, {"output": response})
    
    return jsonify({"response": response})

# Route for generating summary
@app.route('/api/v0/chat_summary', methods=['GET'])
def generate_summary():
    """
    Endpoint to generate a summary of the conversation.
    Returns JSON with 'summary' field containing the generated summary.
    """
    summary = summary_chain.predict(input="Generate a summary")
    return jsonify({"summary": summary})

# Route for getting conversation history
@app.route('/api/v0/chat_/history', methods=['GET'])
def get_history():
    """
    Endpoint to retrieve the conversation history.
    Returns JSON with 'history' field containing the conversation history.
    """
    history = memory.load_memory_variables({})['history']
    return jsonify({"history": history})



@app.route('/')
def root():
    return app.send_static_file('index.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
