import java.util.*;

public class Temp {
    public static void main(String[] args){

    }

    public class Node {
        private int data;

        public int getData() {
            return data;
        }

        public void setData(int data) {
            this.data = data;
        }

        public Node getLink() {
            return link;
        }

        public void setLink(Node link) {
            this.link = link;
        }

        private Node link;
    }
    public static Node reverseList(Node head){
        if(head == null || head.getLink() == null){
            return head;
        }
        Node prev = null;
        while(head != null){
            Node next = head.getLink();
            System.out.println(head);
            head.setLink(prev);
            prev = head;
            head = next;
        }
        return prev;
    }

}
