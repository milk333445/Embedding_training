import pickle
import uuid
import json

file_path = "querydb_20241007.pkl"
with open(file_path, 'rb') as f:
    data = pickle.load(f)
    
save_data = data[["Title", 'Answer']]
output_json = {
    "queries": {},
    "corpus": {},
    "relevant_docs": {},
    "mode": "text"
}
save_data['Title'] = save_data['Title'].str.replace(r'【.*?】', '', regex=True)

for index, row in save_data.iterrows():
    question_uuid = str(uuid.uuid4())
    answer_node = f"node_{index}"
    
    output_json["queries"][question_uuid] = row['Title'].replace("Q：", "").replace("● ", "")
    output_json["corpus"][answer_node] = row['Answer'].replace("A：", "").replace("● ", "").replace("●", "")
    output_json["relevant_docs"][question_uuid] = [answer_node]
    
output_json
with open("qadataset2.json", 'w', encoding='utf-8') as f:
    f.write(json.dumps(output_json, ensure_ascii=False, indent=4))