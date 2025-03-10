class Queue:
    def __init__(self,ms):
        self.queue = list()
        self.msize=ms
    
    
    def toStr(self):
        if len(self.queue)==0:
            return 'E'
            
        positive=sum(self.queue)
        return '_Q:'+str(positive)+'-'+str(len(self.queue))
            
    def clean(self):
        self.queue.clear()
     
    def enqueue(self,data):
        
        self.queue.insert(0,data)
        
        if self.msize>0 and len(self.queue)==self.msize:
            self.queue.pop()
                
    def posScore(self):
        
        if len(self.queue)==0:
            return 0
        return float(sum(self.queue)/len(self.queue))
        
    
    
    
    
    
        