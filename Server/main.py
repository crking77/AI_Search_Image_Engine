from flask import Flask, render_template, url_for, request, send_file, jsonify
from app_embedding.core import embedding_and_result, client,model,collection_name
from qdrant_client.models import PointStruct
from pathlib import Path
from PIL import Image
from uuid import uuid4
BASE_PATH = (Path(__file__).resolve().parent.parent)/"app_embedding/images_data"

app = Flask(__name__)
@app.route("/")
def home_page():
    return  render_template("index.html")
@app.route("/images/<path:filename>")
def get_image(filename):
    return send_file(f"{BASE_PATH}/{filename}")
@app.route("/search_query",methods = ["GET"])
def search_query():
    query_input = request.args.get("search_query")
    result = embedding_and_result(query_input)
    result_path = [(Path(img.payload['path']).name,img.score) for img in result]
    return render_template("Search.html", result =result_path )

@app.route("/update_embedding", methods = ['POST'])
def update_embedding():
    file = request.files.get("file")
    if not file or file.filename =="":
        return jsonify({"error":"No file uploaded"})
    try:
        image = Image.open(file.stream).convert("RGB")
        vector = model.encode(image)
        file_name = Path(file.filename).name
        file_path = BASE_PATH/file_name
        image.save(file_path)
        client.upsert(collection_name=collection_name, points=[PointStruct(id= str(uuid4()),vector = vector.tolist(),payload = {"path": file_path})])
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500
    return render_template("Update.html")



if __name__ == "__main__":
    app.run(debug= False)