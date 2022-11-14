function [period,period_std]=find_T(data,t_interval)
period_vector=[];
for i=1:size(data,1)
    [~,peakpos]=find_peaks(data(i,:),0.1);
    disp(i)
    
    %掐头去尾
    peakpos=peakpos(peakpos~=1&peakpos~=length(data(i,:)));
    peakpos=peakpos(peakpos>3&peakpos<length(data(i,:))-3);
    
    for n=1:length(peakpos)
        left_min=0;right_min=0;
        for j=(peakpos(n)-1):-1:2
            if data(i,j+1)>data(i,j)&&data(i,j-1)>data(i,j)
                left_min=j;
                break;
            end
        end
        if left_min==0
            left_min=1;
        end
        
        for k=(peakpos(n)+1):(size(data,2)-1)
            if data(i,k+1)>data(i,k)&&data(i,k-1)>data(i,k)
                right_min=k;
                break;
            end
        end
        if right_min==0
            right_min=length(data(i,:));
        end
        
        period_vector=[period_vector,t_interval*(right_min-left_min)];
    end
end
period=mean(period_vector);
period_std=std(period_vector);
end