% D = dlmread('process_data.txt',',');
% score=D(:,3)
% score=sort(score);
% d=diff([score;max(score)+1]);
% count = diff(find([1;d])) ;
% y =[score(find(d)) count]
% stem(y,'score 分布','y');axis([0,100,-inf,inf])

% D1= D(:,1);
% d1=diff([D1;max(D1)+1]);
% count1= diff(find([1;d1]));

%----计算每个用户评分平均值
% index=1;
% mean=zeros(length(count1),1);
% for i=1:length(count1)
%     sum=0;
%     for j=index:index+count1(i)-1
%         sum=sum+D3(j);
%     end
%     mean(i)=sum/count1(i);
%     index=index+count1(i);
% end




%%-----------
% count_u=sort(count1);
% d_u=diff([count_u;max(count_u)+1]);
% count3 = diff(find([1;d_u])) ;
% y3 =[count_u(find(d_u)) count3]
% area(y3,'score 分布','y');

%----- 计算每个item 被评分 的平均

% D23=D(:,2:3);
% D2=sort(D23(:,1));
% d2=diff([D2;max(D2)+1]);
% count2= diff(find([1;d2])); %算出 每个item 被几个用户评分

% D23=sortrows(D23,1);

% index=1;
% mean2=zeros(length(count2),1);
% for i=1:length(count2)
%      sum=0;
%      for j=index:index+count2(i)-1
%          sum=sum+D23(j,2);
%      end
%      mean2(i)=sum/count2(i);
%      index=index+count2(i);
% end
% histogram(mean2)
  