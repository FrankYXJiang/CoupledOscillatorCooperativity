%New version of analyzing peak amplitude and phase
function [amp,phase]=amp_phase(data,t_interval,period,minamp)

    activated_data=data(max(data,[],2)>2,:);
    index=1:81;
    phase=[];
    amp=[];
    
    for i=1:size(activated_data,1)
        x=islocalmax(activated_data(i,:));
        loc_localmax=index(x);
        y=islocalmin(activated_data(i,:));
        loc_localmin=index(y);
        
        if loc_localmax(1) < loc_localmin(1)
            loc_localmax(1)=[];
        elseif loc_localmax(end) > loc_localmin(end)
            loc_localmax(end)=[];
        end
        
        %开始计算
        tmp_amp=0.5*(2*activated_data(i,loc_localmax)-activated_data(i,loc_localmin(1:end-1))-activated_data(i,loc_localmin(2:end)));
        
        amp=[amp;tmp_amp(tmp_amp>minamp)];
        loc_localmax=loc_localmax(tmp_amp>minamp);
        tmp_pd=mod((loc_localmax-1)*t_interval,period);
        phase=[phase,tmp_pd];
    end
    
end