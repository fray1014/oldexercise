import java.util.*;
import java.util.regex.Pattern;
public class LeetCode {
    public static void main(String[] args){
        TreeNode treeNode1 = new TreeNode(1);
        TreeNode treeNode2 = new TreeNode(2);
        TreeNode treeNode3 = new TreeNode(3);
        TreeNode treeNode4 = new TreeNode(4);
        TreeNode treeNode5 = new TreeNode(5);
        TreeNode treeNode6 = new TreeNode(6);

        treeNode1.left = treeNode2;
        treeNode1.right = treeNode3;

        treeNode2.left = treeNode4;
        treeNode3.left = treeNode5;
        treeNode3.right = treeNode6;

        SolutionJZ61 serializeTree = new SolutionJZ61();

        String str = serializeTree.Serialize(treeNode1);
        TreeNode treeNode = serializeTree.Deserialize(str);
        System.out.println(str);
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
}


