import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017")
# dbList = client_list_database_names()

myDB = client["myDatabase"]

myCollention = myDB["ai"]

# query = {"$and":[{"name": '송정현', "mobile":"010-8711-9374"}]}
query = {"name": '송정현'}


# for item in myCollention.find({}).sort("name",-1):
#     print(item)

# 삭제방법
for item in myCollention.delete_one(query):
    print(item)

# 결과값 확인
result = myCollention.delete_many(query)

print(result.delete_count)

# 업데이트
newValue = {"$set":{"name":"aaa"}}
result = myCollention.update_one(query, newValue)
print(result.modified_count)
# for item in myCollention.find():
#     print(item)

# # 데이터베이스내의 아이디는 뺴고 출력
# for item in myCollention.find({}, {"_id":0}):
#     print(item)


# for item in myCollention.find({}, {"_id":0, "mobile":0}):
#     print(item)
