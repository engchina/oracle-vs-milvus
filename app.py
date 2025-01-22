import os
import time  # 需要导入time模块用于重试间隔

import gradio as gr
import oracledb
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import connections, utility, MilvusClient

from my_langchain_community.vectorstores import OracleVS
from my_langchain_milvus import Milvus

load_dotenv(find_dotenv())

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_base=os.environ["OPENAI_EMBED_URL"]
)


def load_and_split_documents():
    """从本地文件夹加载并分割 PDF 文档"""
    pdf_files_dir = "files"
    documents = []

    # 遍历指定文件夹中的所有 PDF 文件
    for filename in os.listdir(pdf_files_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(pdf_files_dir, filename)

            # 使用 PyMuPDFLoader 加载 PDF 文件
            loader = TextLoader(file_path)
            documents.extend(loader.load())

    # 使用 RecursiveCharacterTextSplitter 分割文档
    chunk_size = 100
    chunk_overlap = 0
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["。", "！", "？", "，", "\n\n", "\n", ],
        keep_separator=True,
    )
    return text_splitter.split_documents(documents)


def search(query, strategy, top_k):
    documents = load_and_split_documents()
    oracle_result = search_oracle(documents, query, strategy, top_k)
    milvus_result = search_milvus(documents, query, strategy, top_k)
    # chrome_result = search_chrome(documents, query, strategy, top_k)

    # 初始化一个空字典来存储结果
    result_dict = {
        "Oracle Score": [],
        "Oracle Context": [],
        "Milvus Score": [],
        "Milvus Context": [],
        # "Chrome Score": [],
        # "Chrome Context": []
        "Oracle vs Milvus": []  # 用于对比 Oracle Context 和 Milvus Context
    }

    # 添加 Oracle 结果
    for res, score in oracle_result:
        result_dict["Oracle Score"].append(score)
        result_dict["Oracle Context"].append(res.page_content)

    # 添加 Milvus 结果
    for res, score in milvus_result:
        result_dict["Milvus Score"].append(score)
        result_dict["Milvus Context"].append(res.page_content)

    # 对比 Oracle Context 和 Milvus Context 是否相等
    max_length = max(len(oracle_result), len(milvus_result))  # 获取最大长度
    for i in range(max_length):
        oracle_context = oracle_result[i][0].page_content if i < len(oracle_result) else None
        milvus_context = milvus_result[i][0].page_content if i < len(milvus_result) else None
        # 对比逻辑：判断是否相等
        is_equal = oracle_context == milvus_context
        result_dict["Oracle vs Milvus"].append(is_equal)

    # 转换为 Pandas DataFrame
    result_dataframe = pd.DataFrame(result_dict)

    # 如果需要将结果按行对齐，可以使用 fillna 填充缺失值
    result_dataframe = result_dataframe.fillna("")

    return result_dataframe


def search_oracle(documents, query, strategy, top_k):
    try:
        connection = oracledb.connect(dsn=os.environ["ORACLE_23AI_CONNECTION_STRING"])
        print("Connection successful!")

        if strategy == "COSINE":
            oracle_vector_store = OracleVS.from_documents(
                documents,
                embeddings,
                client=connection,
                table_name="Documents_COSINE",
                distance_strategy=DistanceStrategy.COSINE,
            )
        else:
            # VECTOR_DISTANCE with metric EUCLIDEAN is equivalent to L2_DISTANCE:
            oracle_vector_store = OracleVS.from_documents(
                documents,
                embeddings,
                client=connection,
                table_name="Documents_EUCLIDEAN",
                distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
            )

        results = oracle_vector_store.similarity_search_with_score(
            query=query,
            k=top_k,
        )

        print(f"{results=}")

        print("=== 检索到的 Context from Oracle 信息 ===")
        for res, score in results:
            print(f"* [SIM={score:3f}] {res.page_content}")

        return results

    except Exception as e:
        print(f"{e=}")
        print("Connection failed!")


def search_milvus(documents, query, strategy, top_k):
    # The easiest way is to use Milvus Lite where everything is stored in a local file.
    # If you have a Milvus server you can use the server URI such as "http://localhost:19530".
    URI = "http://localhost:19530"
    # 删除集合

    if strategy == "COSINE":
        collection_name = "langchain_cosine_example"

        # 初始化客户端（自动创建本地数据库文件）
        client = MilvusClient(uri=URI)
        if client.has_collection(collection_name):
            client.drop_collection(collection_name)
        client.create_collection(
            collection_name=collection_name,
            dimension=1024,
            metric_type="COSINE"  # 显式指定余弦相似度（默认值，此处明确展示）
        )

        docs = [document.page_content for document in documents]
        # 生成文档向量（使用正确的方法）
        vectors = embeddings.embed_documents(docs)  # 注意这里是 embed_documents
        print(f"{len(vectors[0])=}")

        # 构建插入数据（包含元数据标签）
        data = [{
            "id": i,
            "vector": vec,
            "text": doc,
        } for i, (doc, vec) in enumerate(zip(docs, vectors))]

        # 步骤3：插入数据到集合
        insert_result = client.insert(collection_name, data)
        print(f"插入成功，插入数量：{insert_result['insert_count']}")

        query_vector = embeddings.embed_query(query)  # 注意这里是 embed_query
        print(f"{len(query_vector)=}")

        search_params = {
            "metric_type": "COSINE",
        }

        max_retries = 3
        retry_delay = 1  # 重试间隔时间（秒）
        results = []

        for attempt in range(max_retries):
            try:
                # 执行向量搜索
                searched_results = client.search(
                    collection_name=collection_name,
                    data=[query_vector],
                    limit=top_k,
                    output_fields=["text"],  # 返回文本
                    search_params=search_params,
                    timeout=120,
                )

                # 构造results
                results = []
                for hit in searched_results[0]:
                    score = 1.0 - hit['distance']
                    text_content = hit['entity']['text']
                    res = Document(page_content=text_content)
                    results.append((res, score))

                # 如果有结果则退出重试循环
                if len(results) > 0:
                    break

                # 无结果且是最后一次尝试
                if attempt == max_retries - 1:
                    gr.Error("Milvus 检索出错")
                else:
                    print(f"第 {attempt + 1} 次检索无结果，正在重试...")
                    time.sleep(retry_delay)

            except Exception as e:
                print(f"Milvus 第 {attempt + 1} 次检索失败，错误：{str(e)}")
                if attempt == max_retries - 1:
                    raise gr.Error("Milvus 检索出错，重试次数用尽")
                time.sleep(retry_delay)

        # 最终结果检查
        if not results:
            raise gr.Error("Milvus 检索出错")

        print(f"{len(results)=}")

    else:
        collection_name = "langchain_l2_example"

        connections.connect("default", uri=URI)
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"集合 '{collection_name}' 已删除")
        else:
            print(f"集合 '{collection_name}' 不存在")

        milvus_store = Milvus.from_documents(
            documents,
            embeddings,
            collection_name=collection_name,
            connection_args={"uri": URI},
        )
        results = milvus_store.similarity_search_with_score(
            query=query,
            k=top_k,
        )

    print("=== 检索到的 Context from Milvus 信息 ===")
    for res, score in results:
        print(f"* [SIM={score:3f}] {res.page_content}")

    return results


def search_chrome(documents, query, strategy, top_k):
    # 清除现有的集合（如果存在）
    try:
        chroma_client = Chroma(collection_name="langchain_example")
        chroma_client.delete_collection()
    except Exception as e:
        print(f"清除集合时出错: {e}")

    collection_metadata = {}
    if strategy == "COSINE":
        chrome_store = Chroma.from_documents(
            documents,
            embeddings,
            collection_metadata={"hnsw:space": "cosine"},
            collection_name="langchain_example",

        )
    else:
        chrome_store = Chroma.from_documents(
            documents,
            embeddings,
            collection_name="langchain_example",
        )

    results = chrome_store.similarity_search_with_score(
        query=query,
        k=top_k,
    )

    print("=== 检索到的 Context from Chrome 信息 ===")
    for res, score in results:
        print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")

    return results


with gr.Blocks() as app:
    with gr.Row():
        with gr.Column():
            query_text = gr.Textbox(label="Query")
    with gr.Row():
        with gr.Column():
            strategy_choice = gr.Radio(
                label="Strategy",
                choices=[
                    "L2",
                    "COSINE"
                ],
                value="L2",
            )
        with gr.Column():
            k_slider = gr.Slider(label="Top k", minimum=1, maximum=100, value=10, step=1)

    with gr.Row():
        with gr.Column():
            search_button = gr.Button(
                value="Search",
                variant="primary",
            )

    with gr.Row():
        with gr.Column():
            result_dataframe = gr.Dataframe(
                label="Result",
                show_label=False,
                headers=["Oracle Score",
                         "Oracle Context",
                         "Milvus Score",
                         "Milvus Context",
                         # "Chrome",
                         # "Chrome Context"
                         "Oracle vs Milvus"
                         ],
                column_widths=[30, 70, 30, 70, 50],
                col_count=(5, "fixed"),
                wrap=True,
            )

    search_button.click(
        search,
        inputs=[
            query_text,
            strategy_choice,
            k_slider,
        ],
        outputs=[result_dataframe]
    )

app.queue()

if __name__ == "__main__":
    app.launch()
