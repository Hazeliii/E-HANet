import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class showFlowServlet extends HttpServlet {
    String filePath = "img/";
    List<Flow> flowList1 = new ArrayList<>(), flowList2 = new ArrayList<>();
    String flow_npy, flow_img, events_img;

    @Override
    protected void doPost(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        System.out.println("----showFlowServlet---");
        for(int i=2; i<6;i+=2){
            flow_npy = filePath+String.format("%d_flow_npy.npy", i);
            flow_img = filePath+String.format("inference_%d_flow.png", i);
            events_img = filePath+String.format("inference_%d_events.png", i);
//            System.out.println(flow_img+","+flow_npy);
            Flow flow = new Flow(flow_img, events_img, flow_npy, String.format("%d_flow_npy.npy", i));
            flowList1.add(flow);
        }
        for (Flow flow : flowList1) {
            System.out.println(flow);
        }
        for(int i=6; i<10;i+=2){
            flow_npy = filePath+String.format("%d_flow_npy.npy", i);
            flow_img = filePath+String.format("inference_%d_flow.png", i);
            events_img = filePath+String.format("inference_%d_events.png", i);
//            System.out.println(flow_img+","+flow_npy);
            Flow flow = new Flow(flow_img, events_img, flow_npy, String.format("%d_flow_npy.npy", i));
            flowList2.add(flow);
        }
        req.setAttribute("flowList1", flowList1);
        req.setAttribute("flowList2", flowList2);
        System.out.println("getRequestDispatcher");
        req.getRequestDispatcher("/show.jsp").forward(req, resp);
    }

}
