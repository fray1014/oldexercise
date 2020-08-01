
import java.util.*;
public class Main {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        while (in.hasNextInt()) {// 注意，如果输入是多个测试用例，请通过while循环处理多个测试用例
            int N=in.nextInt();//N个作品
            ArrayList<TreeSet<Integer>> sample=new ArrayList<>(N);
            for(int i=0;i<N;i++){
                int M=in.nextInt();
                TreeSet<Integer> group=new TreeSet<>();
                for(int j=0;j<M;j++){
                    group.add(in.nextInt());
                }
                sample.add(group);
            }
            ArrayList<TreeSet<Integer>> out=result(sample);
            for(int i=0;i<out.size();i++){
                TreeSet<Integer> tmp=out.get(i);
                String outstr="";
                for(int j:tmp){
                    outstr+=j+" ";
                }
                System.out.println(outstr);
            }
        }
    }

    public static ArrayList<TreeSet<Integer>> result(ArrayList<TreeSet<Integer>> sample){
        //ArrayList<TreeSet<Integer>> ret=new ArrayList<>();
        for(int i=0;i<sample.size();i++){
            for(int j=0;j<sample.size();j++){
                if(i==j)
                    continue;
                TreeSet<Integer> tmp1=new TreeSet<Integer>(sample.get(i));
                TreeSet<Integer> tmp2=new TreeSet<Integer>(sample.get(j));
                tmp1.retainAll(tmp2);
                if(tmp1.size()>0){
                    tmp1.addAll(sample.get(i));
                    tmp1.addAll(sample.get(j));
                    sample.set(i,tmp1);
                    sample.remove(j);
                    j--;
                }
            }
        }

        return arrsort(sample);
    }

    public static ArrayList<TreeSet<Integer>> arrsort(ArrayList<TreeSet<Integer>> input){
        for(int i=0;i<input.size()-1;i++){
            for(int j=0;j<input.size()-1-i;j++){
                TreeSet<Integer> tmp1=input.get(j);
                TreeSet<Integer> tmp2=input.get(j+1);
                if(tmp1.first()>tmp2.first()){
                    TreeSet<Integer> tmp=new TreeSet<>(tmp1);
                    input.set(j,tmp2);
                    input.set(j+1,tmp);
                }
            }
        }
        return input;
    }
}
