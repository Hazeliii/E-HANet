import org.apache.commons.fileupload.FileItem;
import org.apache.commons.fileupload.FileItemFactory;
import org.apache.commons.fileupload.FileUploadException;
import org.apache.commons.fileupload.disk.DiskFileItemFactory;
import org.apache.commons.fileupload.servlet.ServletFileUpload;

import javax.servlet.*;
import javax.servlet.http.*;
import java.io.*;
import java.net.URLEncoder;
import java.util.List;

public class uploadFileServlet extends HttpServlet {
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {

    }

    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        /**
         * 处理文件上传
         */
        System.out.println("上传文件中....");
        //1.判断上传的数据是否多段数据
        if(ServletFileUpload.isMultipartContent(request)){
            FileItemFactory fileItemFactory = new DiskFileItemFactory();
            ServletFileUpload servletFileUpload = new ServletFileUpload(fileItemFactory);
            try {
                //解析上传的表单项，得到每一个表单项 fileitem
                List<FileItem> fileItems = servletFileUpload.parseRequest(request);
                for(FileItem item:fileItems){
                    //判断表单项的类型
                    if(item.isFormField()){
                        //普通表单项
                        System.out.println("表单项的name属性值："+item.getFieldName());
                        System.out.println("表单项的value属性值："+item.getString("UTF-8"));
                    }else {
                        //上传的文件
                        System.out.println("表单项的name属性值："+item.getFieldName());
                        String fileName = URLEncoder.encode(item.getName(),"UTF-8");  //解决中文乱码，转换为%XX%XX
                        System.out.println("上传的文件名filename = "+fileName);
                        String savedRoot = "/home/wqm/java_projects/fileTrans/src/main/files/";
                        item.write(new File("/home/wqm/java_projects/fileTrans/src/main/files/"+fileName));
                        //python
                        String filePath = savedRoot+fileName;
                        response.getWriter().write("Successfully saved in "+filePath);
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    protected String runPython(){
        String[] cmds = new String[]{"/home/wqm/anaconda3/envs/eRaft/bin/python",
                "/home/wqm/Projects/E-RAFT-original/E-RAFT-original/estimateForJavaWeb.py", "--visualize"};
        System.out.println("start python");
        String result = null;
        try {
            Process pr = Runtime.getRuntime().exec(cmds);
            final InputStream is1 = pr.getInputStream();
            BufferedReader br = new BufferedReader(new InputStreamReader(is1));
            while ((result = br.readLine())!=null){
                System.out.println(result);
            }
            InputStream is2 = pr.getErrorStream();
            BufferedReader br2 = new BufferedReader(new InputStreamReader(is2));
            String errorMsg = null;
            while ((errorMsg = br2.readLine())!=null){
                System.out.println(errorMsg);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        return result;
    }
}

