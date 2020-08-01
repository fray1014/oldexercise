import org.junit.Test;

import java.util.*;

public class LeetCode2 {
    /**深拷贝复杂链表*/
    public class RandomListNode {
        int label;
        RandomListNode next = null;
        RandomListNode random = null;

        RandomListNode(int label) {
            this.label = label;
        }
    }
    public class SolutionJZ25 {
        public RandomListNode Clone(RandomListNode pHead){
            HashMap<RandomListNode,RandomListNode> hm=new HashMap<>();
            RandomListNode p=pHead;
            while(p!=null){
                RandomListNode tmp=new RandomListNode(p.label);
                hm.put(p,tmp);
                p=p.next;
            }
            p=pHead;
            while(p!=null){
                RandomListNode tmp=hm.get(p);
                tmp.next=p.next==null?null:hm.get(p.next);
                tmp.random=p.random==null?null:hm.get(p.random);
                p=p.next;
            }
            return hm.get(pHead);
        }
    }
    /**输入一颗二叉树的根节点和一个整数，按字典序打印出二叉树中结点值的和为输入整数的所有路径。
     * 路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。*/
    public class TreeNode {
        int val = 0;
        TreeNode left = null;
        TreeNode right = null;

        public TreeNode(int val) {
            this.val = val;

        }
    }
    public class SolutionJZ24 {
        public ArrayList<ArrayList<Integer>> FindPath(TreeNode root,int target) {

            ArrayList<ArrayList<Integer>> result = new ArrayList<>();
            if (root == null){
                return result;
            }


            ArrayList<Integer> path = new ArrayList<>();
            this.find(root, target, result, path);
            return result;
        }


        private void find(TreeNode root, int target, ArrayList<ArrayList<Integer>> result, ArrayList<Integer> path) {
            // 0，当节点为空，return
            if (root == null) {
                return;
            }

            path.add(root.val);
            target -= root.val;

            // 1，当目标值小于0，return
            if(target < 0){
                return;
            }

            // 2，当目标值为0 并且 节点下无其他节点, 保存并返回
            if(target == 0 && root.left == null && root.right == null){
                result.add(path);
                return;
            }

            // 继续遍历左右节点
            // 这里new path是因为左右都会在下次递归path.add(root.val);
            this.find(root.left, target, result, new ArrayList<>(path));
            this.find(root.right, target, result, new ArrayList<>(path));
        }
    }

    /**
     牛牛有一个没有重复元素的数组a，他想要将数组内第n大的数字和第m大的数(从大到小排序)交换位置你能帮帮他吗。
     给定一个数组a，求交换第n大和第m大元素后的数组。*/
    public class Solution0730A{
        public int[] sovle (int[] a, int n, int m) {
            if(n==m||a.length==1)
                return a;
            if(a.length==0||a.length<n||a.length<m)
                return a;
            if(a.length==2){
                int swap=a[0];
                a[0]=a[1];
                a[1]=swap;
            }
            TreeSet<Integer> t=new TreeSet<>();
            for(int i:a){
                t.add(i);
            }
            int index=0;
            int[] b=new int[a.length];
            while(!t.isEmpty()){
                b[index++]=t.last();
                t.pollLast();
            }
            index=0;
            for(int i=0;i<a.length;i++){
                if(index==2)
                    break;
                if(a[i]==b[n-1]){
                    a[i]=b[m-1];
                    index++;
                }
                if(a[i]==b[m-1]){
                    a[i]=b[n-1];
                    index++;
                }
            }
            return a;
        }

    }
    /**输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。
     * 如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。*/
    public class SolutionJZ23 {
        public boolean VerifySquenceOfBST(int [] sequence) {
            if(sequence.length==0)
                return false;
            return helpVerify(sequence,0,sequence.length-1);
        }

        public boolean helpVerify(int[] a,int start,int root){
            if(start>=root)
                return true;
            int i;
            int key=a[root];
            for(i=start;i<root;i++){
                if(a[i]>key)
                    break;
            }
            for(int j=i;j<root;j++){
                if(a[j]<key)
                    return false;
            }
            return helpVerify(a,start,i-1)&&helpVerify(a,i,root-1);
        }
    }
}
