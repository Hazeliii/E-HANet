import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

public class runPython {
    public static void main(String[] args) {
        String[] cmds = new String[]{"/home/wqm/anaconda3/envs/eRaft/bin/python",
                "/home/wqm/Projects/E-RAFT-original/E-RAFT-original/estimateForJavaWeb.py", "--visualize"};
        System.out.println("start python");
        try {
            Process pr = Runtime.getRuntime().exec(cmds);
            final InputStream is1 = pr.getInputStream();
            new Thread(()->{
                BufferedReader br = new BufferedReader(new InputStreamReader(is1));
                try {
                    String result = null;
                    while ((result = br.readLine())!=null){
                        System.out.println(result);
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }).start();
            InputStream is2 = pr.getErrorStream();
            BufferedReader br2 = new BufferedReader(new InputStreamReader(is2));
            String errorMsg = null;
            while ((errorMsg = br2.readLine())!=null){
                System.out.println(errorMsg);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


}
