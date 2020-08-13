import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017")
# dbList = client_list_database_names()

myDB = client["myDatabase"]

myCollention = myDB["ai"]

item = [
    {"name":"송정현", "mobile":"010-8711-9374"},
    {"name":"성준향", "mobile":"011-8711-9374"},
    {"name":"이순신", "mobile":"110-8711-9374"},
    {"name":"파이썬", "mobile":"111-8711-9374"}
]
result = myCollention.insert_many(item)

print(result_inserted_ids)

# if "myDatabase" in dbList:
#     print("나의 데이터베이스가 존재")
# else:
#     print("나의 데이터베이스가 존재하지 않음")