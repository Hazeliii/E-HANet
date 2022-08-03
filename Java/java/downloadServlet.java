import org.apache.commons.io.IOUtils;
import sun.misc.BASE64Encoder;

import javax.servlet.*;
import javax.servlet.http.*;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;

public class downloadServlet extends HttpServlet {
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException{
        System.out.println("----downloadServlet----");
        // 1.获取要下载的文件名
        String name = request.getParameter("name");
        String fileName = "/flows/"+name;
        System.out.println(fileName);

        //2.读取要下载的文件以及文件类型
        ServletContext servletContext = getServletContext();
        String mimeType = servletContext.getMimeType(fileName);
        System.out.println("返回的文件类型："+mimeType);
        InputStream inputStream = servletContext.getResourceAsStream(fileName);

        //3.回传前，通过响应头告诉客户端返回的数据类型
        response.setContentType(mimeType);

        //4.告诉客户端收到的数据适用于下载的（响应头）,不配置的话客户端会直接显示在网页上
        //Content-Disposition:收到的数据如何处理
        // attachment:表示附件，用于下载 ;filename：文件名
        //根据不同的浏览器对文件名进行编码
        String transferredName = name;
        if(request.getHeader("User-Agent").contains("Firefox")){
            System.out.println("火狐");
            response.setHeader("Content-Disposition", "attachment;filename="+"=?utf-8?B?"+
                    new BASE64Encoder().encode(transferredName.getBytes(StandardCharsets.UTF_8))+"?=");
        }else {
            response.setHeader("Content-Disposition", "attachment;filename="+ URLEncoder.encode(transferredName,"UTF-8"));
        }

        //5.把要下载的文件返回给客户端
        //将输入流copy给输出流
        OutputStream outputStream = response.getOutputStream();
        IOUtils.copy(inputStream,outputStream);
    }
}
