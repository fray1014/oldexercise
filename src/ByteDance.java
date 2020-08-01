import java.util.*;
public class ByteDance {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        while (in.hasNextLine()) {// 注意，如果输入是多个测试用例，请通过while循环处理多个测试用例
            String time=in.nextLine();
            System.out.println(totalSecond(time));
        }
    }

    public static String totalSecond(String time){
        //1970/1/1 0:0:0
        String[] s=time.split(" ");
        if(s.length!=2)
            return "bad param";
        String[] date=s[0].split("/");
        if(date.length!=3)
            return "bad param";

        int year=Integer.parseInt(date[0]);
        if(year<1970)
            return "bad param";
        boolean isRun=isRunYear(year);

        int month=Integer.parseInt(date[1]);
        if(month<1||month>12)
            return "bad param";
        int day=Integer.parseInt(date[2]);
        if(month==2){
            if(isRun) {
                if (day > 29 || day < 1)
                    return "bad param";
            }else if(day>28||day<1)
                return "bad param";
        }else if(is31days(month)){
            if(day>31||day<1)
                return "bad param";
        }else{
            if(day>30||day<1)
                return "bad param";
        }

        String[] hms=s[1].split(":");
        if(hms.length!=3)
            return "bad param";
        int hour=Integer.parseInt(hms[0]);
        int minute=Integer.parseInt(hms[1]);
        int second=Integer.parseInt(hms[2]);
        if(hour>23||hour<0)
            return "bad param";
        if(minute>59||minute<0)
            return "bad param";
        if(second>59||second<0)
            return "bad param";
        //输入合法判断完毕
        long sum=helpFindSeconds(year,month,day,hour,minute,second);
        return String.valueOf(sum);
    }

    public static boolean is31days(int month){
        if(month==1||month==3||month==5||month==7||month==8
        ||month==10||month==12)
            return true;
        else
            return false;
    }

    public static boolean isRunYear(int year){

        if(year%4==0&&year%100!=0||year%400==0)
            return true;
        return false;
    }

    public static long helpFindSeconds(int year,int month,int day,int hour,int minute,int second){
        boolean isThisYearRun=isRunYear(year);
        long sum=0;
        sum+=second;

        sum+=minute*60;

        sum+=hour*3600;
        if(day>0)
            sum+=(day-1)*24*3600;
        if(month>1){
            for(int i=1;i<month;i++){
                if(is31days(i))
                    sum+=31*24*3600;
                else if(i==2){
                    sum+=28*24*3600;
                    if(isThisYearRun)
                        sum+=1*24*3600;
                }else{
                    sum+=30*24*3600;
                }
            }
        }

        for(int i=1970;i<year;i++){
            boolean tmpb=isRunYear(i);
            if(tmpb){
                sum+=366*24*3600;
            }else{
                sum+=365*24*3600;
            }
        }

        return sum;
    }
}
