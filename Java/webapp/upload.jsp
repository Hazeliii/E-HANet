<%--
  Created by IntelliJ IDEA.
  User: wqm
  Date: 2022/8/2
  Time: 上午10:05
  To change this template use File | Settings | File Templates.
--%>
<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<html>
<head>
    <title>uploadFile</title>
</head>
<body>
<form action="http://localhost:8080/fileTrans/upload" method="post" enctype="multipart/form-data">
    用户名：<input type="text" name="username"> <br/>
    头像：<input type="file" name="file"> <br/>
    <input type="submit" value="上传">
</form>
</body>
</html>
