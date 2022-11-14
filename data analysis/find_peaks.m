function [peaks,peak_pos]=find_peaks(data,delta)
%note the data here is only one single cell.
    peaks=[];
    peak_pos=[];
    %first find the local maximum and minimum.
    
    if_maxmin=zeros(1,length(data));
    if_maxmin(1)=1; if_maxmin(end)=1;
    
    for i=2:length(data)-1
        if (data(i)-data(i-1))*(data(i)-data(i+1))>0
            if_maxmin(i)=1;
        end
    end
    maxmin_pos=find(if_maxmin==1);
    
    count=1;
    for i=2:length(if_maxmin)-1
        if if_maxmin(i)==1
            count=count+1;
            if data(maxmin_pos(count))-data(maxmin_pos(count-1))>delta && data(maxmin_pos(count))-data(maxmin_pos(count+1))>delta
                peaks=[peaks,data(maxmin_pos(count))];
                peak_pos=[peak_pos,maxmin_pos(count)];
            end     
        end
    end
end