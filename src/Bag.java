public class Bag {
    public static int knapSack(int[] w, int[] v, int C){
        int size=w.length;
        if(size==0){
            return 0;
        }
        int[] dp=new int[C+1];
        for(int i=0;i<=C;i++){
            dp[0]=w[0]<=i?v[0]:0;
        }
        for(int i=1;i<size;i++){
            for(int j=C;j>=w[i];j--){
                dp[j]= Math.max(dp[j],v[i]+dp[j-w[i]]);
            }
        }
        return dp[C];
    }
}
