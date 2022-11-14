function pd=find_pds(data,period,t_interval)

    pd=[];
    
    for i=1:size(data,1)
        [~,peak_pos]=find_peaks(data(i,:),0.2);
        peak_pos=peak_pos*t_interval-t_interval;
        pd=[pd,mod(peak_pos,period)];
    end
    pd=pd/period*2*pi;
    
end