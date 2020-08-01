import org.junit.Test;

import java.util.Arrays;

public class SortTemp {
    @Test
    public void test(){
        int[] a={1};
        for(int i:MergeSort(a)){
            System.out.println(i);
        }
    }

    public static int[] MergeSort(int[] input){
        if(input.length<2)
            return input;
        int mid=input.length/2;
        int[] left= Arrays.copyOfRange(input,0,mid);
        int[] right=Arrays.copyOfRange(input,mid,input.length);
        return merge(MergeSort(left),MergeSort(right));
    }

    public static int[] merge(int[] left,int[] right){
        int[] ret=new int[left.length+right.length];
        for(int index=0,i=0,j=0;index<ret.length;index++){
            if(i>=left.length){
                ret[index]=right[j++];
            }else if(j>=right.length){
                ret[index]=left[i++];
            }else if(left[i]>right[j]){
                ret[index]=right[j++];
            }else{
                ret[index]=left[i++];
            }
        }
        return ret;
    }
}
