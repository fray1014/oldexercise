import org.junit.Test;

import java.util.*;
import java.util.regex.Pattern;
public class LeetCode {
    public static void main(String[] args){
        System.out.println(TestStatic.test());
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
    public static class ListNode{
        int val;
        ListNode next=null;
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
    public static class SolutionJZ51{
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
    public static class SolutionJZ48{
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
    public static class TreeNode {
        int val = 0;
        TreeNode left = null;
        TreeNode right = null;

        public TreeNode(int val) {
            this.val = val;
        }
    }

    public static class SolutionJZ38 {
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
    public static class SolutionJZ5{
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
    public static class SolutionJZ67{
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
    public static class SolutionJZ63{
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

    /**
     * 二叉树镜像*/
    public static class SolutionJZ18{
        public void Mirror(TreeNode root) {
            if(root==null) {
                return;
            }
            TreeNode tmp=root.left;
            root.left=root.right;
            root.right=tmp;
            Mirror(root.left);
            Mirror(root.right);

        }
    }

    /**一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。
     * 求该青蛙跳上一个n级的台阶总共有多少种跳法。*/
    public static class SolutionJZ9{
        public int JumpFloorII(int target) {
            if(target==0||target==1) return 1;
            int sum=1;
            for(int i=2;i<=target;i++){
                sum<<=1;
            }
            return sum;
        }
    }

    /**从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行*/
    public static class SolutionJZ60 {
        ArrayList<ArrayList<Integer> > Print(TreeNode pRoot) {
            ArrayList<ArrayList<Integer>> thelist = new ArrayList<ArrayList<Integer>>();
            if(pRoot==null)return thelist; //这里要求返回thelist而不是null
            Queue<TreeNode> q=new LinkedList<TreeNode>();
            q.offer(pRoot);
            while(!q.isEmpty()){
                ArrayList<Integer> list=new ArrayList<Integer>();
                int s=q.size();
                for(int i=0;i<s;i++){
                    TreeNode tmp=q.poll();
                    list.add(tmp.val);
                    if(tmp.left!=null)
                        q.offer(tmp.left);
                    if(tmp.right!=null)
                        q.offer(tmp.right);
                }
                thelist.add(list);
            }
            return thelist;
        }

    }

    /**给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。
     * 注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。
     *
     * 1.有右子树，下一结点是右子树中的最左结点，例如 B，下一结点是 H
     *
     * 2.无右子树，且结点是该结点父结点的左子树，则下一结点是该结点的父结点，例如 H，下一结点是 E
     *
     * 3.无右子树，且结点是该结点父结点的右子树，则我们一直沿着父结点追朔，直到找到某个结点是其父结点的左子树，
     * 如果存在这样的结点，那么这个结点的父结点就是我们要找的下一结点。例如 I，下一结点是 A；
     * 例如 G，并没有符合情况的结点，所以 G 没有下一结点*/
    public class TreeLinkNode {
        int val;
        TreeLinkNode left = null;
        TreeLinkNode right = null;
        TreeLinkNode next = null;

        TreeLinkNode(int val) {
            this.val = val;
        }
    }

    public static class SolutionJZ57 {

        public TreeLinkNode GetNext(TreeLinkNode pNode) {
            // 1.
            if (pNode.right != null) {
                TreeLinkNode pRight = pNode.right;
                while (pRight.left != null) {
                    pRight = pRight.left;
                }
                return pRight;
            }
            // 2.
            if (pNode.next != null && pNode.next.left == pNode) {
                return pNode.next;
            }
            // 3.
            if (pNode.next != null) {
                TreeLinkNode pNext = pNode.next;
                while (pNext.next != null && pNext.next.right == pNext) {
                    pNext = pNext.next;
                }
                return pNext.next;
            }
            return null;
        }
    }

    /**给一个链表，若其中包含环，请找出该链表的环的入口结点，否则，输出null。*/
    public static class SolutionJZ55{
        public ListNode EntryNodeOfLoop(ListNode pHead){
            if(pHead == null || pHead.next == null){
                return null;
            }

            ListNode fast = pHead;
            ListNode slow = pHead;

            while(fast != null && fast.next != null){
                fast = fast.next.next;
                slow = slow.next;
                if(fast == slow){
                    ListNode slow2 = pHead;
                    while(slow2 != slow){
                        slow2 = slow2.next;
                        slow = slow.next;
                    }
                    return slow2;
                }
            }
            return null;

        }
    }

    /**请实现一个函数用来找出字符流中第一个只出现一次的字符。例如，当从字符流中只读出前两个字符"go"时，第一个只出现一次的字符是"g"。
     * 当从该字符流中读出前六个字符“google"时，第一个只出现一次的字符是"l"。*/
    public static class SolutionJZ54{
        Queue<Character> q=new LinkedList<Character>();
        HashMap<Character,Integer> hm=new HashMap<Character,Integer>();
        //Insert one char from stringstream
        public void Insert(char ch){
            if(hm.containsKey(ch)){
                int val=hm.get(ch)+1;
                hm.put(ch,val);
            }else{
                hm.put(ch,1);
                q.offer(ch);
            }
        }
        //return the first appearence once char in current stringstream
        public char FirstAppearingOnce(){
            while(!q.isEmpty()){
                if(hm.get(q.peek())>1){
                    q.poll();
                }else{
                    return q.peek();
                }
            }
            return '#';
        }
    }

    /***请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。
     * 例如，字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。
     * 但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。
     */

    public static class SolutionJZ53 {
        public static boolean isNumeric(String str) {
            String pattern = "^[-+]?\\d*(?:\\.\\d*)?(?:[eE][+\\-]?\\d+)?$";
            String pa="^[-+]?\\d*(\\.\\d*)?[A-z]*$";
            String s = new String(str);
            return Pattern.matches(pattern,s);//Pattern.matches(pattern,s);
        }
    }

    /**地上有一个m行和n列的方格。一个机器人从坐标0,0的格子开始移动，每一次只能向左，右，上，下四个方向移动一格，
     * 但是不能进入行坐标和列坐标的数位之和大于k的格子。
     * 例如，当k为18时，机器人能够进入方格（35,37），因为3+5+3+7 = 18。
     * 但是，它不能进入方格（35,38），因为3+5+3+8 = 19。请问该机器人能够达到多少个格子？*/
    public static class SolutionJZ66{
        public int movingCount(int threshold, int rows, int cols) {
            if (rows <= 0 || cols <= 0 || threshold < 0)
                return 0;

            boolean[][] isVisited = new boolean[rows][cols];//标记
            int count = movingCountCore(threshold, rows, cols, 0, 0, isVisited);
            return count;
        }

        private int movingCountCore(int threshold,int rows,int cols,
                                    int row,int col, boolean[][] isVisited) {
            if (row < 0 || col < 0 || row >= rows || col >= cols || isVisited[row][col]
                    || cal(row) + cal(col) > threshold)
                return 0;
            isVisited[row][col] = true;
            return 1 + movingCountCore(threshold, rows, cols, row - 1, col, isVisited)
                    + movingCountCore(threshold, rows, cols, row + 1, col, isVisited)
                    + movingCountCore(threshold, rows, cols, row, col - 1, isVisited)
                    + movingCountCore(threshold, rows, cols, row, col + 1, isVisited);
        }

        private int cal(int num) {
            int sum = 0;
            while (num > 0) {
                sum += num % 10;
                num /= 10;
            }
            return sum;
        }
    }
    /**请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。
     * 路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，向下移动一个格子。*/
    public static class SolutionJZ65{
        boolean[] visited = null;
        public boolean hasPath(char[] matrix, int rows, int cols, char[] str) {
            visited = new boolean[matrix.length];
            for(int i = 0; i < rows; i++)
                for(int j = 0; j < cols; j++)
                    if(subHasPath(matrix,rows,cols,str,i,j,0))
                        return true;
            return false;
        }

        public boolean subHasPath(char[] matrix, int rows, int cols, char[] str, int row, int col, int len){
            if(matrix[row*cols+col] != str[len]|| visited[row*cols+col] == true) return false;
            if(len == str.length-1) return true;
            visited[row*cols+col] = true;
            if(row > 0 && subHasPath(matrix,rows,cols,str,row-1,col,len+1)) return true;
            if(row < rows-1 && subHasPath(matrix,rows,cols,str,row+1,col,len+1)) return true;
            if(col > 0 && subHasPath(matrix,rows,cols,str,row,col-1,len+1)) return true;
            if(col < cols-1 && subHasPath(matrix,rows,cols,str,row,col+1,len+1)) return true;
            visited[row*cols+col] = false;
            return false;
        }

        /*public int[][] visit;
        public boolean hasPath(char[] matrix, int rows, int cols, char[] str) {
            visit = new int[rows][cols];
            char[][] array = new char[rows][cols];
            for (int i = 0; i < rows ; i++) {
                for(int j = 0; j < cols; j++) {
                    array[i][j] = matrix[i*cols + j];
                }
            }
            for (int i = 0; i < rows ; i++) {
                for(int j = 0; j < cols; j++) {
                    if(find(array,rows,cols,str,i,j,0)){
                        return  true;
                    }
                }
            }
            return false;
        }
        public boolean find(char[][] array, int rows, int cols, char[] str, int rpos,int cpos, int spos) {

            if(spos >= str.length) {
                return  true;
            }
            if(rpos < 0 || cpos < 0 || rpos >= rows || cpos >= cols || array[rpos][cpos] != str[spos] || visit[rpos][cpos] == 1) {

                return false;
            }
            visit[rpos][cpos] = 1;
            boolean isSunc =  find( array,   rows,  cols, str,  rpos+1, cpos, spos+1)
                    || find( array,   rows,  cols, str,  rpos , cpos+1, spos+1)
                    || find( array,   rows,  cols, str,  rpos-1, cpos, spos+1)
                    || find( array,   rows,  cols, str,  rpos , cpos-1, spos+1);
            visit[rpos][cpos] = 0;
            return isSunc;
        }*/

    }

    /**给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值*/
    public static class SolutionJZ64{
        public ArrayList<Integer> maxInWindows(int [] num, int size){
            ArrayList<Integer> max=new ArrayList<Integer>();
            if(num.length==0 || size<1 || size>num.length) {
                return max;
            }
            for(int i=0;i<num.length-size+1;i++){
                int[] tmp=getArray(num,size,i);
                max.add(getMax(tmp));
            }
            return max;
        }
        public int[] getArray(int[] a,int size,int index){
            int[] b=new int[size];
            for(int i=0;i<size;i++){
                b[i]=a[index+i];
            }
            return b;
        }
        public int getMax(int[] a){
            int max=a[0];
            for(int i=1;i<a.length;i++){
                if(a[i]>max)
                    max=a[i];
            }
            return max;
        }
    }

    /**序列化和反序列化二叉树*/
    public static class SolutionJZ61{
        /*
        String Serialize(TreeNode root) {
            if (root == null) return "";
            return helpSerialize(root, new StringBuilder()).toString();
        }

        private StringBuilder helpSerialize(TreeNode root, StringBuilder s) {
            if (root == null) return s;
            s.append(root.val).append("!");
            if (root.left != null) {
                helpSerialize(root.left, s);
            } else {
                s.append("#!"); // 为null的话直接添加即可
            }
            if (root.right != null) {
                helpSerialize(root.right, s);
            } else {
                s.append("#!");
            }
            return s;
        }
        private int index = 0; // 设置全局主要是遇到了#号的时候需要直接前进并返回null

        TreeNode Deserialize(String str) {
            if (str == null || str.length() == 0) return null;
            String[] split = str.split("!");
            return helpDeserialize(split);
        }

        private TreeNode helpDeserialize(String[] strings) {
            if (strings[index].equals("#")) {
                index++;// 数据前进
                return null;
            }
            // 当前值作为节点已经被用
            TreeNode root = new TreeNode(Integer.parseInt(strings[index]));
            index++; // index++到达下一个需要反序列化的值
            root.left = helpDeserialize(strings);
            root.right = helpDeserialize(strings);
            return root;
        }*/

        int index = -1;
        /**
         * 分别遍历左节点和右节点，空使用#代替，节点之间，隔开
         *
         * @param root
         * @return
         */
        public String Serialize(TreeNode root) {
            if (root == null) {
                return "#";
            } else {
                return root.val + "," + Serialize(root.left) + "," + Serialize(root.right);
            }
        }
        /**
         * 使用index来设置树节点的val值，递归遍历左节点和右节点，如果值是#则表示是空节点，直接返回
         *
         * @param str
         * @return
         */
        TreeNode Deserialize(String str) {
            String[] s = str.split(",");//将序列化之后的序列用，分隔符转化为数组
            index++;//索引每次加一
            int len = s.length;
            if (index > len) {
                return null;
            }
            TreeNode treeNode = null;
            if (!s[index].equals("#")) {//不是叶子节点 继续走 是叶子节点出递归
                treeNode = new TreeNode(Integer.parseInt(s[index]));
                treeNode.left = Deserialize(str);
                treeNode.right = Deserialize(str);
            }
            return treeNode;
        }
    }

    /**
     * 给定一棵二叉搜索树，请找出其中的第k小的结点。
     * 例如（5，3，7，2，4，6，8）中，按结点数值大小顺序第三小结点的值为4。*/
    public static class SolutionJZ62{
        ArrayList<TreeNode> list = new ArrayList<>();
        TreeNode KthNode(TreeNode pRoot, int k){
            addNote(pRoot);
            if(k>=1&&list.size()>=k){
                return list.get(k-1);
            }
            return null;
        }
        //中序遍历
        void addNote(TreeNode cur){
            if(cur!=null){
                addNote(cur.left);
                list.add(cur);
                addNote(cur.right);
            }
        }
    }

    /**请实现一个函数按照之字形打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右至左的顺序打印，
     * 第三行按照从左到右的顺序打印，其他行以此类推。*/
    public static class SolutionJZ59{
        public ArrayList<ArrayList<Integer> > Print(TreeNode pRoot) {
            int cnt=1;
            ArrayList<ArrayList<Integer>> thelist = new ArrayList<ArrayList<Integer>>();
            if(pRoot==null)return thelist; //这里要求返回thelist而不是null
            //奇数行
            Stack<TreeNode> q=new Stack<>();
            //偶数行
            Stack<TreeNode> st=new Stack<>();
            q.push(pRoot);
            while(!q.isEmpty()||!st.isEmpty()){
                ArrayList<Integer> list=new ArrayList<Integer>();
                if(cnt%2==1){
                    int s=q.size();
                    for(int i=0;i<s;i++){
                        TreeNode tmp=q.pop();
                        list.add(tmp.val);
                        if(tmp.left!=null)
                            st.push(tmp.left);
                        if(tmp.right!=null)
                            st.push(tmp.right);
                    }
                }else{
                    int s=st.size();
                    for(int i=0;i<s;i++){
                        TreeNode tmp=st.pop();
                        list.add(tmp.val);
                        if(tmp.right!=null)
                            q.push(tmp.right);
                        if(tmp.left!=null)
                            q.push(tmp.left);
                    }
                }

                cnt++;
                thelist.add(list);
            }
            return thelist;
        }
    }

    /**请实现一个函数，用来判断一棵二叉树是不是对称的。
     * 注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。*/
    public static class SolutionJZ58{
        boolean isSymmetrical(TreeNode pRoot){
            return pRoot==null||jude(pRoot.left,pRoot.right);
        }

        public boolean jude(TreeNode node1, TreeNode node2) {
            if (node1 == null && node2 == null) {
                return true;
            } else if (node1 == null || node2 == null) {
                return false;
            }

            if (node1.val != node2.val) {
                return false;
            } else {
                return jude(node1.left, node2.right) && jude(node1.right, node2.left);
            }
        }
    }

    /**在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。
     * 例如，链表1->2->3->3->4->4->5 处理后为 1->2->5*/
    public static class SolutionJZ56{
        public ListNode deleteDuplication(ListNode pHead){
            if(pHead == null || pHead.next == null){
                return pHead;
            }
            // 自己构建辅助头结点
            ListNode head=new ListNode(Integer.MIN_VALUE);
            head.next = pHead;
            ListNode pre = head;
            ListNode cur = head.next;
            while(cur!=null){
                if(cur.next != null && cur.next.val == cur.val){
                    // 相同结点一直前进
                    while(cur.next != null && cur.next.val == cur.val){
                        cur = cur.next;
                    }
                    // 退出循环时，cur 指向重复值，也需要删除，而 cur.next 指向第一个不重复的值
                    // cur 继续前进
                    cur = cur.next;
                    // pre 连接新结点
                    pre.next = cur;
                }else{
                    pre = cur;
                    cur = cur.next;
                }
            }
            return head.next;
        }
    }

    /**在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，但不知道有几个数字是重复的。
     * 也不知道每个数字重复几次。请找出数组中任意一个重复的数字。
     * 例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。*/
    public static class SolutionJZ50{
        public boolean duplicate(int numbers[], int length, int [] duplication) {
            if (numbers == null || length == 0) {
                return false;
            }
            for (int i = 0; i < length; i++) {
                while (numbers[i] != i) {
                    if (numbers[i] == numbers[numbers[i]]) {
                        duplication[0] = numbers[i];
                        return true;
                    }
                    else {
                        int temp = numbers[i];
                        numbers[i] = numbers[temp];
                        numbers[temp] = temp;
                    }
                }
            }
            return false;
        }
    }

    /**请实现一个函数用来匹配包括'.'和'*'的正则表达式。模式中的字符'.'表示任意一个字符，
     * 而'*'表示它前面的字符可以出现任意次（包含0次）。
     * 在本题中，匹配是指字符串的所有字符匹配整个模式。
     * 例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配*/
    public static class SolutionJZ52{
        public boolean matchStr(char[] str, int i, char[] pattern, int j) {

            // 边界
            if (i == str.length && j == pattern.length) { // 字符串和模式串都为空
                return true;
            } else if (j == pattern.length) { // 模式串为空
                return false;
            }

            boolean flag = false;
            boolean next = (j + 1 < pattern.length && pattern[j + 1] == '*'); // 模式串下一个字符是'*'
            if (next) {
                if (i < str.length && (pattern[j] == '.' || str[i] == pattern[j])) { // 要保证i<str.length，否则越界
                    return matchStr(str, i, pattern, j + 2) || matchStr(str, i + 1, pattern, j);
                } else {
                    return matchStr(str, i, pattern, j + 2);
                }
            } else {
                if (i < str.length && (pattern[j] == '.' || str[i] == pattern[j])) {
                    return matchStr(str, i + 1, pattern, j + 1);
                } else {
                    return false;
                }
            }
        }

        public boolean match(char[] str, char[] pattern) {
            return matchStr(str, 0, pattern, 0);
        }
    }

    /**将一个字符串转换成一个整数，要求不能使用字符串转换整数的库函数。
     * 数值为0或者字符串不是一个合法的数值则返回0*/
    public static class SolutionJZ49{

        public static int StrToInt(String str) {
            /*偷懒解法，异常捕获
            Integer res=0;
            try {
                res = new Integer(str);
            } catch (NumberFormatException e) {

            } finally {
                return res;
            }
        }*/

            //最优解
            if(str == null || "".equals(str.trim()))return 0;
            str = str.trim();
            char[] arr = str.toCharArray();
            int i = 0;
            int flag = 1;
            int res = 0;
            if(arr[i] == '-'){
                flag = -1;
            }
            if( arr[i] == '+' || arr[i] == '-'){
                i++;
            }
            while(i<arr.length ){
                //是数字
                if(isNum(arr[i])){
                    int cur = arr[i] - '0';
                    if(flag == 1 && (res > Integer.MAX_VALUE/10 || res == Integer.MAX_VALUE/10 && cur >7)){
                        return 0;
                    }
                    if(flag == -1 && (res > Integer.MAX_VALUE/10 || res == Integer.MAX_VALUE/10 && cur >8)){
                        return 0;
                    }
                    res = res*10 +cur;
                    i++;
                }else{
                    //不是数字
                    return 0;
                }
            }
            return res*flag;
        }
        public static boolean isNum(char c){
            return c>='0'&& c<='9';
        }
    }

    /**求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。*/
    public static class SolutionJZ47{
        int sum=0;
        public int Sum_Solution(int n) {
            //递归法
            /*
            int sum=n;
            boolean ans = (n>0)&&((sum+=Sum_Solution(n-1))>0);
            return sum;*/
            //公式法
            int a=n+(int)Math.pow(n,2);
            return a>>1;
        }
    }

    /**每年六一儿童节,牛客都会准备一些小礼物去看望孤儿院的小朋友,今年亦是如此。
     * HF作为牛客的资深元老,自然也准备了一些小游戏。其中,有个游戏是这样的:
     * 首先,让小朋友们围成一个大圈。然后,他随机指定一个数m,让编号为0的小朋友开始报数。
     * 每次喊到m-1的那个小朋友要出列唱首歌,然后可以在礼品箱中任意的挑选礼物,并且不再回到圈中,
     * 从他的下一个小朋友开始,继续0...m-1报数....这样下去....直到剩下最后一个小朋友,可以不用表演,
     * 并且拿到牛客名贵的“名侦探柯南”典藏版(名额有限哦!!^_^)。
     * 请你试着想下,哪个小朋友会得到这份礼品呢？(注：小朋友的编号是从0到n-1)

     如果没有小朋友，请返回-1*/
    public static class SolutionJZ46{
        public int LastRemaining_Solution(int n, int m) {
            //1.链表模拟，时间复杂度O(N^2)，空间复杂度O(N)
            /*
            if(n<1)
                return -1;
            else{
                LinkedList<Integer> l=new LinkedList<>();
                for(int i=0;i<n;i++){
                    l.offer(i);
                }
                int index=0;
                while(n>1){
                    index=(index+m-1)%n;
                    l.remove(index);
                    n--;
                }
                return l.peek();
            }*/
            //2.迭代
            if (n <= 0) return -1;
            int index = 0;
            for (int i=2; i<=n; ++i) {
                index = (index + m) % i;
            }
            return index;
        }
    }

    /**LL今天心情特别好,因为他去买了一副扑克牌,发现里面居然有2个大王,2个小王(一副牌原本是54张^_^)...
     * 他随机从中抽出了5张牌,想测测自己的手气,看看能不能抽到顺子,如果抽到的话,
     * 他决定去买体育彩票,嘿嘿！！“红心A,黑桃3,小王,大王,方片5”,“Oh My God!”不是顺子.....LL不高兴了,
     * 他想了想,决定大\小 王可以看成任何数字,并且A看作1,J为11,Q为12,K为13。
     * 上面的5张牌就可以变成“1,2,3,4,5”(大小王分别看作2和4),“So Lucky!”。LL决定去买体育彩票啦。
     * 现在,要求你使用这幅牌模拟上面的过程,然后告诉我们LL的运气如何， 如果牌能组成顺子就输出true，
     * 否则就输出false。为了方便起见,你可以认为大小王是0。*/
    public static class SolutionJZ45{
        public boolean isContinuous(int [] numbers){
            if(numbers.length!=5){
                return false;
            }else{
                TreeSet<Integer> s=new TreeSet<>();
                int cnt=0;
                for(int i=0;i<numbers.length;i++){
                    if(numbers[i]==0){
                        cnt++;
                    }else{
                        s.add(numbers[i]);
                    }
                }
                if(cnt+s.size()!=5){
                    return false;
                }else if(s.last()-s.first()<5){
                    return true;
                }
                return false;
            }
        }
    }

    /**牛客最近来了一个新员工Fish，每天早晨总是会拿着一本英文杂志，写些句子在本子上。同事Cat对Fish写的内容颇感兴趣，
     * 有一天他向Fish借来翻看，但却读不懂它的意思。例如，“student. a am I”。
     * 后来才意识到，这家伙原来把句子单词的顺序翻转了，正确的句子应该是“I am a student.”。
     * Cat对一一的翻转这些单词顺序可不在行，你能帮助他么？*/
    public static class SolutionJZ44{
        public static String ReverseSentence(String str){
            if(str==null)
                return str;
            String[] s=str.split(" ");
            if(s.length==0)
                return str;
            Stack<String> tmp=new Stack<>();
            for(int i=0;i<s.length;i++){
                tmp.push(s[i]);
            }
            String ret=new String();
            while(!tmp.isEmpty()){
                ret+=tmp.pop();
                ret+=" ";
            }
            return ret.trim();
        }
    }

    /**汇编语言中有一种移位指令叫做循环左移（ROL），现在有个简单的任务，就是用字符串模拟这个指令的运算结果。
     * 对于一个给定的字符序列S，请你把其循环左移K位后的序列输出。
     * 例如，字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”。是不是很简单？OK，搞定它！*/
    public static class SolutionJZ43{
        public String LeftRotateString(String str,int n){
            //队列
            /*
            if(str==null || n<=0 || str.length()<n)
                return str;
            String ret=new String();
            Queue<Character> q=new LinkedList<Character>();
            char[] s=str.toCharArray();
            for(int i=0;i<s.length;i++){
                q.offer(s[i]);
            }
            for(int i=0;i<n;i++){
                q.offer(q.poll());
            }
            for (Character c:q
            ) {
                ret+=c;
            }
            return ret;*/
            if (str == null || n > str.length()) {
                return str;
            }
            return str.substring(n) + str.substring(0, n);
        }
    }

    /**输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，
     * 如果有多对数字的和等于S，输出两个数的乘积最小的。*/
    public static class SolutionJZ42{
        public ArrayList<Integer> FindNumbersWithSum(int [] array,int sum) {
            if(array.length==0)
                return new ArrayList<Integer>();
            int i=0;
            ArrayList<Integer> a=new ArrayList<Integer>();
            int j=array.length-1;
            while(i<j){
                if(array[i]+array[j]==sum){
                    a.add(array[i]);
                    a.add(array[j]);
                    break;
                }else if(array[i]+array[j]>sum){
                    j--;
                }else{
                    i++;
                }

            }
            return a;
        }
    }

    /**小明很喜欢数学,有一天他在做数学作业时,要求计算出9~16的和,他马上就写出了正确答案是100。
     * 但是他并不满足于此,他在想究竟有多少种连续的正数序列的和为100(至少包括两个数)。
     * 没多久,他就得到另一组连续正数和为100的序列:18,19,20,21,22。
     * 现在把问题交给你,你能不能也很快的找出所有和为S的连续正数序列?*/
    public static class SolutionJZ41{
        public ArrayList<ArrayList<Integer> > FindContinuousSequence(int sum) {
            ArrayList<ArrayList<Integer> > ret=new ArrayList<ArrayList<Integer> >();
            int l=1,r=2;
            int tmp=1;
            while(l<=sum/2){
                if(tmp<sum){
                    tmp+=r;
                    r++;
                }else if(tmp>sum){
                    tmp-=l;
                    l++;
                }else{
                    ArrayList<Integer> a=new ArrayList<>();
                    for(int i=l;i<r;i++){
                        a.add(i);
                    }
                    ret.add(a);
                    tmp-=l;
                    l++;
                }
            }
            return ret;
        }
    }

    /**一个整型数组里除了两个数字之外，其他的数字都出现了两次。
     * 请写程序找出这两个只出现一次的数字。*/
    public static class SolutionJZ40{
        public void FindNumsAppearOnce(int [] array,int num1[] , int num2[]) {
            HashMap<Integer,Integer> h=new HashMap<>();
            for (int i:array
                 ) {
                if(!h.containsKey(i)){
                    h.put(i,1);
                }else{
                    h.put(i,2);
                }
            }
            int cnt=0;
            for (int i:array
                 ) {
                if(h.get(i)==1){
                    if(cnt==0) {
                        num1[0] = i;
                        cnt++;
                    }else{
                        num2[0]=i;
                    }
                }
            }
        }
    }

    /**输入一棵二叉树，判断该二叉树是否是平衡二叉树。

     在这里，我们只需要考虑其平衡性，不需要考虑其是不是排序二叉树*/
    public static class SolutionJZ39{
        public boolean IsBalanced_Solution(TreeNode root) {
            if(root==null){
                return true;
            }
            if(Math.abs(TreeDepth(root.left)-TreeDepth(root.right))>1){
                return false;
            }
            return IsBalanced_Solution(root.left)&&IsBalanced_Solution(root.right);
        }

        public int TreeDepth(TreeNode root) {
            if(root==null){
                return 0;
            }
            if(root.left==null && root.right==null){
                return 1;
            }
            return 1+Math.max(TreeDepth(root.left),TreeDepth(root.right));
        }
    }

    /**统计一个数字在排序数组中出现的次数。*/
    public static class SolutionJZ37{
        public int GetNumberOfK(int [] array , int k) {
            int index = Arrays.binarySearch(array, k);
            if(index<0)return 0;
            int cnt = 1;
            for(int i=index+1; i < array.length && array[i]==k;i++)
                cnt++;
            for(int i=index-1; i >= 0 && array[i]==k;i--)
                cnt++;
            return cnt;
            /*
            if(array.length == 0 || k < array[0] || k > array[array.length-1]){
                return 0;
            }
            int left = 0;
            int right = array.length -1;
            int count = 0;
            int found = 0;
            int mid = -1;
            while(left < right){
                mid = (left+right)/2;
                if(array[mid] > k){
                    right = mid-1;
                }else if(array[mid] < k){
                    left = mid+1;
                }else{
                    count++;
                    found = mid;
                    break;
                }
            }

            int prev = mid-1;
            int foll = mid+1;
            while(prev >= left){
                if(array[prev] == k){
                    count++;
                    prev--;
                }else{
                    break;
                }
            }

            while(foll <= right){
                if(array[foll] == k){
                    count++;
                    foll++;
                }else{
                    break;
                }
            }
            return count;*/
        }
    }

    /**找两个链表的第一个公共节点*/
    public static class SolutionJZ36{
        public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
            if(pHead1 == null || pHead2 == null)return null;
            ListNode p1 = pHead1;
            ListNode p2 = pHead2;
            while(p1!=p2){
                p1 = p1.next;
                p2 = p2.next;
                if(p1 != p2){
                    if(p1 == null)p1 = pHead2;
                    if(p2 == null)p2 = pHead1;
                }
            }
            return p1;
        }
    }

    /**找逆序对*/
    public static class SolutionJZ35{
        private int cnt;
        private void MergeSort(int[] array, int start, int end){
            if(start>=end)return;
            int mid = (start+end)/2;
            MergeSort(array, start, mid);
            MergeSort(array, mid+1, end);
            MergeOne(array, start, mid, end);
        }
        private void MergeOne(int[] array, int start, int mid, int end){
            int[] temp = new int[end-start+1];
            int k=0,i=start,j=mid+1;
            while(i<=mid && j<= end){
//如果前面的元素小于后面的不能构成逆序对
                if(array[i] <= array[j])
                    temp[k++] = array[i++];
                else{
//如果前面的元素大于后面的，那么在前面元素之后的元素都能和后面的元素构成逆序对
                    temp[k++] = array[j++];
                    cnt = (cnt + (mid-i+1))%1000000007;
                }
            }
            while(i<= mid)
                temp[k++] = array[i++];
            while(j<=end)
                temp[k++] = array[j++];
            for(int l=0; l<k; l++){
                array[start+l] = temp[l];
            }
        }
        public int InversePairs(int [] array) {
            MergeSort(array, 0, array.length-1);
            return cnt;
        }
    }

    /**第一个出现一次的字符串*/
    public static class SolutionJZ34{
        public int FirstNotRepeatingChar(String str) {

            if(str==null)
                return -1;

            int[] cnt=new int[256];
            for(int i=0;i<str.length();i++)
                cnt[str.charAt(i)]++;
            for(int i=0;i<str.length();i++){
                if(cnt[str.charAt(i)]==1)
                    return i;
            }
            return -1;
            /*
            HashMap<Character,Integer> hm=new HashMap<>();
            for(int i=0;i<str.length();i++){
                if(!hm.containsKey(str.charAt(i))){
                    hm.put(str.charAt(i),1);
                }else{
                    int tmp=hm.get(str.charAt(i));
                    hm.put(str.charAt(i),++tmp);
                }
            }
            for(int i=0;i<str.length();i++){
                if(hm.get(str.charAt(i))==1){
                    return i;
                }
            }
            return -1;*/
        }
    }

    /**丑数
     * 把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含质因子7。
     * 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。*/
    public static class SolutionJZ33{
        public int GetUglyNumber_Solution(int index) {
            if(index <= 0)return 0;
            int p2=0,p3=0,p5=0;//初始化三个指向三个潜在成为最小丑数的位置
            int[] result = new int[index];
            result[0] = 1;//
            for(int i=1; i < index; i++){
                result[i] = Math.min(result[p2]*2, Math.min(result[p3]*3, result[p5]*5));
                if(result[i] == result[p2]*2)p2++;//为了防止重复需要三个if都能够走到
                if(result[i] == result[p3]*3)p3++;//为了防止重复需要三个if都能够走到
                if(result[i] == result[p5]*5)p5++;//为了防止重复需要三个if都能够走到
            }
            return result[index-1];
        }
    }

    /**单链表逆序*/
    public static ListNode reverse(ListNode head){
        if(head==null || head.next==null)
            return head;
        ListNode prev=null;
        while(head!=null){
            ListNode tmp=head.next;
            head.next=prev;
            prev=head;
            head=tmp;
        }
        return prev;
    }

    /**输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。
     * 例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。*/

    public static class SolutionJZ32{
        public String PrintMinNumber(int [] numbers) {

            if(numbers == null || numbers.length == 0)return "";
            for(int i=0; i < numbers.length; i++){
                for(int j = i+1; j < numbers.length; j++){
                    int sum1 = Integer.valueOf(numbers[i]+""+numbers[j]);
                    int sum2 = Integer.valueOf(numbers[j]+""+numbers[i]);
                    if(sum1 > sum2){
                        int temp = numbers[j];
                        numbers[j] = numbers[i];
                        numbers[i] = temp;
                    }
                }
            }
            String str = new String("");
            for(int i=0; i < numbers.length; i++)
                str = str + numbers[i];
            return str;

        }
    }

    public static int DP(int n){

        //递归
        if(n==0)
            return 0;
        if(n==1){
            return 1;
        }
        if(n==2){
            return 2;
        }
        return DP(n-1)+DP(n-2);

        //动态规划
        /*
        int[] a=new int[n+1];
        a[0]=0;
        a[1]=1;
        a[2]=2;
        for(int i=3;i<=n;i++){
            a[i]=a[i-1]+a[i-2];
        }
        return a[n];*/
    }

    /**背包问题（动态规划）
     * F(i,C)=max(F(i−1,C),v(i)+F(i−1,C−w(i)))*/
    public static class KnapSack01 {
        public static int knapSack(int[] w, int[] v, int C) {
            int size = w.length;
            if (size == 0) {
                return 0;
            }

            int[] dp = new int[C + 1];
            //初始化第一行
            //仅考虑容量为C的背包放第0个物品的情况
            for (int i = 0; i <= C; i++) {
                dp[i] = w[0] <= i ? v[0] : 0;
            }

            for (int i = 1; i < size; i++) {
                for (int j = C; j >= w[i]; j--) {
                    dp[j] = Math.max(dp[j], v[i] + dp[j - w[i]]);
                }
            }
            return dp[C];
        }

        /*
        public static void main(String[] args) {
            int[] w = {2, 1, 3, 2};
            int[] v = {12, 10, 20, 15};
            System.out.println(knapSack(w, v, 5));
        }*/
    }

    /**1~n整数1的个数*/
    public static class SolutionJZ31{
        public int NumberOf1Between1AndN_Solution(int n) {
            if(n==0){
                return 0;
            }
            int cnt=0;
            /*
            return NumberOf1Between1AndN_Solution(n-1)+NumberOf1(n);*/
            for(int i=1;i<=n;i++){
                cnt+=NumberOf1(i);
            }
            return cnt;
        }
         public int NumberOf1(int n){
            String str=Integer.toString(n);
            char[] ch=str.toCharArray();
            int cnt=0;
            for(char c:ch){
                if(c=='1'){
                    cnt++;
                }
            }
            return cnt;
         }
         //从位数上求1
         /*
            int count=0;
            for(int i=n;i>0;i--){
                for(int j=i;j>0;j/=10){
                    if(j%10==1) count++;
                }
            }
            return count;*/
    }

    /**连续子向量的最大和*/
    public static class SolutionJZ30{
        public int FindGreatestSumOfSubArray(int[] array) {

            int max = array[0];
            for (int i = 1; i < array.length; i++) {
                array[i] += array[i - 1] > 0 ? array[i - 1] : 0;
                max = Math.max(max, array[i]);
            }
            return max;

        }
    }

    /**最小的K个数*/
    public static class SolutionJZ29{
        public ArrayList<Integer> GetLeastNumbers_Solution(int [] input, int k) {
            ArrayList<Integer> list = new ArrayList<>();
            if (input == null || input.length == 0 || k > input.length || k == 0)
                return list;
            /*
            int[] arr = new int[k + 1];//数组下标0的位置作为哨兵，不存储数据
            //初始化数组
            for (int i = 1; i < k + 1; i++)
                arr[i] = input[i - 1];
            buildMaxHeap(arr, k + 1);//构造大根堆
            for (int i = k; i < input.length; i++) {
                if (input[i] < arr[1]) {
                    arr[1] = input[i];
                    adjustDown(arr, 1, k + 1);//将改变了根节点的二叉树继续调整为大根堆
                }
            }
            for (int i = 1; i < arr.length; i++) {
                list.add(arr[i]);
            }
            return list;*/
            TreeSet<Integer> t=new TreeSet<>();
            for(int i:input){
                t.add(i);
            }
            for(int i=0;i<k;i++){
                list.add(t.first());
                t.pollFirst();
            }
            return list;
        }
        /**
         * @Author: ZwZ
         * @Description: 构造大根堆
         * @Param: [arr, length]  length:数组长度 作为是否跳出循环的条件
         * @return: void
         * @Date: 2020/1/30-22:06
         */
        public void buildMaxHeap(int[] arr, int length) {
            if (arr == null || arr.length == 0 || arr.length == 1)
                return;
            for (int i = (length - 1) / 2; i > 0; i--) {
                adjustDown(arr, i, arr.length);
            }
        }
        /**
         * @Author: ZwZ
         * @Description: 堆排序中对一个子二叉树进行堆排序
         * @Param: [arr, k, length]
         * @return:
         * @Date: 2020/1/30-21:55
         */
        public void adjustDown(int[] arr, int k, int length) {
            arr[0] = arr[k];//哨兵
            for (int i = 2 * k; i <= length; i *= 2) {
                if (i < length - 1 && arr[i] < arr[i + 1])
                    i++;//取k较大的子结点的下标
                if (i > length - 1 || arr[0] >= arr[i])
                    break;
                else {
                    arr[k] = arr[i];
                    k = i; //向下筛选
                }
            }
            arr[k] = arr[0];
        }
    }

    /**数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
     * 例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，
     * 因此输出2。如果不存在则输出0。*/
    public static class SolutionJZ28{
        public int MoreThanHalfNum_Solution(int [] array) {

            if(array==null)
                return 0;
            if(array.length==1)
                 return array[0];
            HashMap<Integer,Integer> hm=new HashMap<>();
            for(int i:array){
                if(!hm.containsKey(i)){
                    hm.put(i,1);
                }else{
                    int tmp=hm.get(i);
                    tmp++;
                    if(tmp>array.length/2){
                        return i;
                    }
                    hm.put(i,tmp);
                }
            }
            return 0;
            /*
            if(array == null || array.length == 0)return 0;
            int preValue = array[0];//用来记录上一次的记录
            int count = 1;//preValue出现的次数（相减之后）
            for(int i = 1; i < array.length; i++){
                if(array[i] == preValue)
                    count++;
                else{
                    count--;
                    if(count == 0){
                        preValue = array[i];
                        count = 1;
                    }
                }
            }
            int num = 0;//需要判断是否真的是大于1半数，这一步骤是非常有必要的，因为我们的上一次遍历只是保证如果存在超过一半的数就是preValue，但不代表preValue一定会超过一半
            for(int i=0; i < array.length; i++)
                if(array[i] == preValue)
                    num++;
            return (num > array.length/2)?preValue:0;*/

        }
    }

    /**字符串排列*/
    public static class SolutionJZ27{

        public ArrayList<String> PermutationHelp(StringBuilder str){

            ArrayList<String> result = new  ArrayList<String>();
            if(str.length() == 1)result.add(str.toString());
            else{
                for(int i = 0; i < str.length(); i++){
                    if(i== 0  || str.charAt(i) != str.charAt(0)){
                        char temp = str.charAt(i);
                        str.setCharAt(i, str.charAt(0));
                        str.setCharAt(0, temp);
                        ArrayList<String> newResult = PermutationHelp(new StringBuilder(str.substring(1)));
                        for(int j =0; j < newResult.size(); j++)
                            result.add(str.substring(0,1)+newResult.get(j));
                        //用完还是要放回去的
                        temp = str.charAt(0);
                        str.setCharAt(0, str.charAt(i));
                        str.setCharAt(i, temp);
                    }
                }
                //需要在做一个排序操作

            }
            return result;
        }

        public ArrayList<String> Permutation(String str) {
            StringBuilder strBuilder = new StringBuilder(str);
            ArrayList<String> result = PermutationHelp(strBuilder);
            return result;
        }
        /*
        ArrayList<String> a=new ArrayList<>();
        public ArrayList<String> Permutation(String str) {

            if(str==null||str.length()<0){
                return a;
            }
            char[] c=str.toCharArray();
            Permutation(c,0);
            return a;
        }
        public void Permutation(char[] c,int index){
            if(index>=c.length){
                a.add(c.toString());
            }else{
                for(int i=index;i<c.length;++i){
                    char tmp=c[i];
                    c[i]=c[index];
                    c[index]=tmp;
                    Permutation(c,i+1);
                    tmp=c[i];
                    c[i]=c[index];
                    c[index]=tmp;
                }
            }
        }*/
    }
    /**斐波那契数列*/
    public class SolutionJZ7 {
        public int Fibonacci(int n) {
            if(n<=0)
                return 0;
            if(n==1)
                return 1;
            return Fibonacci(n-1)+Fibonacci(n-2);
        }
    }

    /**输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。
     * 要求不能创建任何新的结点，只能调整树中结点指针的指向。*/
    public class SolutionJZ26{
        public TreeNode Convert(TreeNode pRootOfTree) {
            if(pRootOfTree == null){
                return null;
            }
            ArrayList<TreeNode> list = new ArrayList<>();
            Convert(pRootOfTree, list);
            return Convert(list);
        }

        public void Convert(TreeNode pRootOfTree, ArrayList<TreeNode> list){
            if(pRootOfTree.left != null){
                Convert(pRootOfTree.left, list);
            }

            list.add(pRootOfTree);

            if(pRootOfTree.right != null){
                Convert(pRootOfTree.right, list);
            }
        }

        public TreeNode Convert(ArrayList<TreeNode> list){
            for(int i = 0; i < list.size() - 1; i++){
                list.get(i).right = list.get(i + 1);
                list.get(i + 1).left = list.get(i);
            }
            return list.get(0);
        }
    }
    @Test
    public void test(){
        System.out.println(new BigData().add("10","20"));
    }

}


