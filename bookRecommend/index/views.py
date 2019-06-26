#-*-coding:utf-8-*-
from django.shortcuts import render,HttpResponseRedirect,HttpResponse,render_to_response
from login.views import connect,close
#from basedUserCF import adjustrecommend
#-*- coding:utf-8-*-
# Create your views here.

#猜你喜欢模块
def index(request,stu_id,grade):
    #连接数据库
    db,cursor = connect()
    #转换成gbk编码
    #namegbk = name.encode("gbk")
    #namegbk=name
    #获取当前用户的id
    #sql_getid = "select stu_id from user where username='"+stu_id+"' and "
    #cursor.execute(sql_getid)
    #for row in cursor.fetchall():
     #   uid = row[0]
    # print uid
    #获取推荐结果
    #bookid_list = adjustrecommend(uid)
    #
    stu=stu_id
    gra=grade
    sql_getreco= "select item_ids from rank where stu_id='"+stu+"' and stu_grade='"+gra+"'"
    #bookid_list=[]
    cursor.execute(sql_getreco)
    for row in cursor.fetchall():
        #bookid_list.append(row[0])
        bookid_list=row[0].split(",")

    # print bookid_list
    bookdic = {}
    #定义sql，提交并查询
    for bid in bookid_list:
        sql = "select * from book where item_id = '"+bid+"'"
        #sql = "select * from recommend where username='" + namegbk + "'"
        cursor.execute(sql)
        for row in cursor.fetchall():
            #bookdic[row[1]] = {"bname":row[0].decode("gbk"),"bdisnum":row[2],"bscore":row[3]}
            bookdic[row[0]]={"bname":row[1],"category":row[2]}
    close(db,cursor)
    #返回语句，带回相应的数据
    return  render_to_response("index.html",{
        "bookdic":bookdic,
        "stu_id":stu_id,
        "grade": grade,
        "color1": "red",
        "flag":True,
    })

def hot(request,stu_id,grade):
    booklist = []
    #连接数据库
    db,cursor = connect()
    #定义sql，提交并查询
    sql = "select * from book"
    cursor.execute(sql)
    for row in cursor.fetchall():
        #booklist.append({"bname":row[0].decode("gbk"),"bid":row[1],"bdisnum":row[2],"bscore":row[3]})
        booklist.append({"bname": row[1], "bid": row[0],"category": row[2]})
    #排序函数
    booklist = sorted(booklist,reverse=True)
    newbooklist = []
    i = 0;
    for one in booklist:
        if i<15:
            newbooklist.append(one)
            i +=1
    print i
    #返回语句，带回相应的数据
    return  render_to_response("index.html", {
        "booklist":newbooklist,
        #"name":name,
        "stu_id":stu_id,
        "grade": grade,
        "color2": "red",
        "flag":False,
    })