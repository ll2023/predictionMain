import numpy

class Hist:
    def __init__(self,bs=10):
        self.input=list()
        self.ends=list()
        self.bins=bs
        self.intervals=list()
        
    
    def addItem(self,fv,pred):
        self.input.append((fv,pred))
        v = round(fv,2)
        if v not in self.ends:
            self.ends.append(v)
            
    def getHist(self):
        start = min(self.ends)
        end = max(self.ends)
        step = (end-start)/self.bins
        
        while start<end-step:
            self.intervals.append((start,start+step))
            start=start+step
        
        histogram={}
        total=0
        for (v,p) in self.input:
            total = total+max(p,0)
            for i in range(0,len(self.intervals)):
                (st,en) = self.intervals[i]
                if v>=st and v<en:
                    histogram[i] = histogram[i] + max(p,0)
        
        if total>0:
            for k in histogram.keys():
                histogram[k] = histogram[k] / total
        
        return histogram
            
            
            
            
        
        
    
    
        
        

    
    
        
    
    
    
    
    
        