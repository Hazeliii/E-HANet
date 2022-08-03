<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core" %>
<%--
  Created by IntelliJ IDEA.
  User: wqm
  Date: 2022/8/2
  Time: 下午9:01
  To change this template use File | Settings | File Templates.
--%>
<%@ page contentType="text/html;charset=UTF-8" language="java" isELIgnored="false" %>
<link type="text/css" rel="stylesheet" href="css/style.css" >
<html>
<head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <!-- The above 3 meta tags *must* come first in the head; any other head content must come *after* these tags -->
    <meta name="description" content="">
    <meta name="author" content="">
    <!--    <link rel="icon" href="../../favicon.ico">-->

    <title>Show Flow</title>

    <script type="text/javascript">
        $(function (){
            $("a.download").click(function (){
                alert("sssd")
            })
        })
    </script>

    <!-- Bootstrap core CSS -->
    <link href="css/bootstrap.min.css" rel="stylesheet">

    <!-- IE10 viewport hack for Surface/desktop Windows 8 bug -->
    <link href="css/ie10-viewport-bug-workaround.css" rel="stylesheet">

    <!-- Custom styles for this template -->
    <link href="cover.css" rel="stylesheet">

    <!-- Just for debugging purposes. Don't actually copy these 2 lines! -->
    <!--[if lt IE 9]><script src="../../assets/js/ie8-responsive-file-warning.js"></script>[endif]-->
    <!--    <script src="../../assets/js/ie-emulation-modes-warning.js"></script>-->

    <!-- HTML5 shim and Respond.js for IE8 support of HTML5 elements and media queries -->
    <!--[if lt IE 9]>
    <script src="https://oss.maxcdn.com/html5shiv/3.7.3/html5shiv.min.js"></script>
    <script src="https://oss.maxcdn.com/respond/1.4.2/respond.min.js"></script>
    <![endif]-->


</head>

<body>

<div>
    <div style="margin-bottom: 100px">
        <div class="inner" style="margin-left: 200px; margin-right: 200px">
            <h3 class="masthead-brand">Show Flow</h3>
            <nav>
                <ul class="nav masthead-nav">
                    <li ><a href="http://localhost:8080/fileTrans">Home</a></li>
                    <li class="active"><a href="#">Show</a></li>
                    <li><a href="#">Contact</a></li>
                </ul>
            </nav>
        </div>
    </div>

    <table class="img_table">
    <tr>
        <c:forEach items="${requestScope.flowList1}" var="flow">
            <td width="700px" style="margin: 10px">
                <img class="book_img" src="${flow.eventPath}">
                <img class="book_img" src="${flow.flowPath}">
            </td>
        </c:forEach>
    </tr>

    <tr style="margin-top: 100px">
            <c:forEach items="${requestScope.flowList1}" var="flow">
                <td width="700px" style="margin-top: 10px; margin-bottom: 10px" align="center">
                    <a class="download" href="http://localhost:8080/fileTrans/download?name=${flow.npyName}" style="color: #d9edf7"> Download ${flow.npyName}</a>
                </td>
            </c:forEach>
     </tr>
    </table>

    <div style="margin-bottom: 100px">

    </div>

    <table class="img_table">
        <tr>
            <c:forEach items="${requestScope.flowList2}" var="flow">
                <td width="700px" style="margin: 10px">
                    <img class="book_img" src="${flow.eventPath}">
                    <img class="book_img" src="${flow.flowPath}">
                </td>
            </c:forEach>
        </tr>

        <tr style="margin-top: 100px">
            <c:forEach items="${requestScope.flowList2}" var="flow">
                <td width="700px" style="margin-top: 10px; margin-bottom: 10px" align="center">
                    <a class="download"  style="color: #d9edf7" href="http://localhost:8080/fileTrans/download?name=${flow.npyName}"> Download ${flow.npyName}</a>
                </td>
            </c:forEach>
        </tr>
    </table>
</div>

<!-- Bootstrap core JavaScript
================================================== -->
<!-- Placed at the end of the document so the pages load faster -->
<script src="https://code.jquery.com/jquery-1.12.4.min.js" integrity="sha384-nvAa0+6Qg9clwYCGGPpDQLVpLNn0fRaROjHqs13t4Ggj3Ez50XnGQqc/r8MhnRDZ" crossorigin="anonymous"></script>
<script>window.jQuery || document.write('<script src="js/jquery-3.2.1.min.js"><\/script>')</script>
<script src="js/bootstrap.min.js"></script>
<!-- IE10 viewport hack for Surface/desktop Windows 8 bug -->
<script src=js/ie10-viewport-bug-workaround.js"></script>
</body>
</html>
