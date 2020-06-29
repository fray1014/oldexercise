import java.util.*;

public class LeetCode {
    public static void main(String[] args){

    }
    /*寻找两数之和（两遍哈希表）*/
    //给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。
    public static int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            map.put(nums[i], i);
        }
        for (int i = 0; i < nums.length; i++) {
            int complement = target - nums[i];
            if (map.containsKey(complement) && map.get(complement) != i) {
                return new int[] { i, map.get(complement) };
            }
        }
        throw new IllegalArgumentException("No two sum solution");
    }
    /*两数相加*/
    //给出两个 非空 的链表用来表示两个非负的整数。其中，它们各自的位数是按照 逆序 的方式存储的，
    // 并且它们的每个节点只能存储 一位 数字。
    //如果，我们将这两个数相加起来，则会返回一个新的链表来表示它们的和。
    public class ListNode{
        int val;
        ListNode next;
        ListNode(int x){val=x;}
    }
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode dummyHead = new ListNode(0);
        ListNode p = l1, q = l2, curr = dummyHead;
        int carry = 0;
        while (p != null || q != null) {
            int x = (p != null) ? p.val : 0;
            int y = (q != null) ? q.val : 0;
            int sum = carry + x + y;
            carry = sum / 10;
            curr.next = new ListNode(sum % 10);
            curr = curr.next;
            if (p != null) p = p.next;
            if (q != null) q = q.next;
        }
        if (carry > 0) {
            curr.next = new ListNode(carry);
        }
        return dummyHead.next;
    }
    /*寻找最长回文子串*/
    //时间复杂度：O(n²）O(n²）
    //空间复杂度：O(1）O(1）
    public String longestPalindrome(String s) {
        if (s == null || s.length() < 1) return "";
        int start = 0, end = 0;
        for (int i = 0; i < s.length(); i++) {
            int len1 = expandAroundCenter(s, i, i);
            int len2 = expandAroundCenter(s, i, i + 1);
            int len = Math.max(len1, len2);
            if (len > end - start) {
                start = i - (len - 1) / 2;
                end = i + len / 2;
            }
        }
        return s.substring(start, end + 1);
    }

    private int expandAroundCenter(String s, int left, int right) {
        int L = left, R = right;
        while (L >= 0 && R < s.length() && s.charAt(L) == s.charAt(R)) {
            L--;
            R++;
        }
        return R - L - 1;
    }
    /*字符串转整数*/
    //首先，该函数会根据需要丢弃无用的开头空格字符，直到寻找到第一个非空格的字符为止。
    //
    //当我们寻找到的第一个非空字符为正或者负号时，则将该符号与之后面尽可能多的连续数字组合起来，作为该整数的正负号；假如第一个非空字符是数字，则直接将其与之后连续的数字字符组合起来，形成整数。
    //
    //该字符串除了有效的整数部分之后也可能会存在多余的字符，这些字符可以被忽略，它们对于函数不应该造成影响。
    //
    //注意：假如该字符串中的第一个非空格字符不是一个有效整数字符、字符串为空或字符串仅包含空白字符时，则你的函数不需要进行转换。
    //
    //在任何情况下，若函数不能进行有效的转换时，请返回 0。
    public int myAtoi(String str) {
        str = str.trim();
        if (str == null || str.length() == 0)
            return 0;

        char firstChar = str.charAt(0);
        int sign = 1;
        int start = 0;
        long res = 0;
        if (firstChar == '+') {
            sign = 1;
            start++;
        } else if (firstChar == '-') {
            sign = -1;
            start++;
        }

        for (int i = start; i < str.length(); i++) {
            if (!Character.isDigit(str.charAt(i))) {
                return (int) res * sign;
            }
            res = res * 10 + str.charAt(i) - '0';
            if (sign == 1 && res > Integer.MAX_VALUE)
                return Integer.MAX_VALUE;
            if (sign == -1 && res > Integer.MAX_VALUE)
                return Integer.MIN_VALUE;
        }
        return (int) res * sign;
    }

    /**
     * 单词拆分*/
    public static class Solution0 {
    /*
        动态规划算法，dp[i]表示s前i个字符能否拆分
        转移方程：dp[j] = dp[i] && check(s[i+1, j]);
        check(s[i+1, j])就是判断i+1到j这一段字符是否能够拆分
        其实，调整遍历顺序，这等价于s[i+1, j]是否是wordDict中的元素
        这个举个例子就很容易理解。
        假如wordDict=["apple", "pen", "code"],s = "applepencode";
        dp[8] = dp[5] + check("pen")
        翻译一下：前八位能否拆分取决于前五位能否拆分，加上五到八位是否属于字典
        （注意：i的顺序是从j-1 -> 0哦~
    */

        public static HashMap<String, Boolean> hash = new HashMap<>();
        public static boolean wordBreak(String s, List<String> wordDict) {
            boolean[] dp = new boolean[s.length()+1];

            //方便check，构建一个哈希表
            for(String word : wordDict){
                hash.put(word, true);
            }

            //初始化
            dp[0] = true;

            //遍历
            for(int j = 1; j <= s.length(); j++){
                for(int i = j-1; i >= 0; i--){
                    dp[j] = dp[i] && check(s.substring(i, j));
                    if(dp[j])   break;
                }
            }

            return dp[s.length()];
        }

        public static boolean check(String s){
            return hash.getOrDefault(s, false);
        }
    }

    /**
     * 给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。
     * 不能使用除法。（注意：规定B[0] = A[1] * A[2] * ... * A[n-1]，B[n-1] = A[0] * A[1] * ... * A[n-2];）*/
    public static class Solution1{
        public static int[] multiply(int[] A){
            int[] B=new int[A.length];
            B[0]=1;
            for(int i=1;i<A.length;i++){
                B[i]=B[i-1]*A[i-1];
            }
            int tmp=1;
            for(int j=A.length-2;j>=0;j--){
                tmp*=A[j+1];
                B[j]*=tmp;
            }
            return B;
        }
    }

    /**
     * 写一个函数，求两个整数之和，要求在函数体内不得使用+、-、*、/四则运算符号。
     */
    public static class Solution2{
        public static int Add(int num1,int num2) {
            while(num2!=0){
                int tmp=(num1 & num2)<<1;
                num1^=num2;
                num2=tmp;
            }
            return num1;
        }
    }

    /**
     * 输入二叉树求深度
     */
    public class TreeNode {
        int val = 0;
        TreeNode left = null;
        TreeNode right = null;

        public TreeNode(int val) {
            this.val = val;
        }
    }

    public static class Solution3 {
        public static int TreeDepth(TreeNode root) {
            if(root==null) {
                return 0;
            }
            int leftDeepth=TreeDepth(root.left);
            int rightDeepth=TreeDepth(root.right);
            return Math.max(leftDeepth,rightDeepth)+1;
        }
    }

    /**
     * 用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。*/
    public static class Solution4{
        Stack<Integer> stack1 = new Stack<Integer>();
        Stack<Integer> stack2 = new Stack<Integer>();

        public void push(int node) {
            stack1.push(node);
        }

        public int pop() {
            if(stack2.empty()){
                while(!stack1.empty()){
                    stack2.push(stack1.peek());
                    stack1.pop();
                }
            }
            int ret = stack2.peek();
            stack2.pop();
            return ret;
        }
    }

    /**
     * 给你一根长度为n的绳子，请把绳子剪成整数长的m段（m、n都是整数，n>1并且m>1，m<=n），
     * 每段绳子的长度记为k[1],...,k[m]。请问k[1]x...xk[m]可能的最大乘积是多少？
     * 例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。*/
    public static class Solution5{
        public int cutRope(int target) {
           if(target<=0) return 0;
                if(target==1 || target == 2) return 1;
                if(target==3) return 2;
                int m = target % 3;
                switch(m){
                    case 0 :
                        return (int) Math.pow(3, target / 3);
                    case 1 :
                        return (int) Math.pow(3, target / 3 - 1) * 4;
                    case 2 :
                        return (int) Math.pow(3, target / 3) * 2;
                }
                return 0;
        }
    }

    /**
     * 得到一个数据流中的中位数*/
    public static class Solution6{
        //小顶堆，用该堆记录位于中位数后面的部分
        private PriorityQueue<Integer> minHeap = new PriorityQueue<Integer>();

        //大顶堆，用该堆记录位于中位数前面的部分
        private PriorityQueue<Integer> maxHeap = new PriorityQueue<Integer>(15, new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return o2 - o1;
            }
        });

        //记录偶数个还是奇数个
        int count = 0;
        //每次插入小顶堆的是当前大顶堆中最大的数
        //每次插入大顶堆的是当前小顶堆中最小的数
        //这样保证小顶堆中的数永远大于等于大顶堆中的数
        //中位数就可以方便地从两者的根结点中获取了
        //优先队列中的常用方法有：增加元素，删除栈顶，获得栈顶元素，和队列中的几个函数应该是一样的
        //offer peek poll,
        public void Insert(Integer num) {
            //个数为偶数的话，则先插入到大顶堆，然后将大顶堆中最大的数插入小顶堆中
            if(count % 2 == 0){
                maxHeap.offer(num);
                int max = maxHeap.poll();
                minHeap.offer(max);
            }else{
                //个数为奇数的话，则先插入到小顶堆，然后将小顶堆中最小的数插入大顶堆中
                minHeap.offer(num);
                int min = minHeap.poll();
                maxHeap.offer(min);
            }
            count++;
        }
        public Double GetMedian() {
            //当前为偶数个，则取小顶堆和大顶堆的堆顶元素求平均
            if(count % 2 == 0){
                return new Double(minHeap.peek() + maxHeap.peek())/2;
            }else{
                //当前为奇数个，则直接从小顶堆中取元素即可，所以我们要保证小顶堆中的元素的个数。
                return new Double(minHeap.peek());
            }
        }
    }
}


