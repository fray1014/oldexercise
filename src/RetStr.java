import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;

public class RetStr {
    @Test
    public void test(){
        listRetStr("qssb?qssbbssq");
    }

    public ArrayList<String> listRetStr(String input){
        ArrayList<String> arr=new ArrayList<>();
        char[] in=input.toCharArray();
        int start=0;
        int end=0;
        for(int i=0;i<in.length;i++){
            start=i;
            for(int j=in.length-1;j>i;j--){
                end=j;
                while(i<j){
                    if(in[i]==in[j]){
                        i++;
                        j--;
                    }else{
                        break;
                    }
                }
                if(i>=j){
                    char[] ctmp= Arrays.copyOfRange(in,start,end+1);
                    String tmp=new String(ctmp);
                    arr.add(tmp);
                    System.out.println(tmp);
                }
                i=start;
                j=end;
            }
        }
        return arr;
    }
}
