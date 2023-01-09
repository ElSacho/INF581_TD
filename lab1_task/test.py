def move(y, tabY, T=3):
        if len(y) == T:
            tabY.append(y.copy())
            y.pop()
            return 
        
        y_ = range(1,4)
        
        for deplacement in y_:
            l = len(y)
            y.append(deplacement)
            print(y)
            move(y,tabY ,T)
            y = y[:l]
            
def gen_All_path2(T=3):
        tabY = []
        
        for y_1 in [1,2]:
            move([y_1],tabY,T)
        print(tabY)
        return tabY
    
gen_All_path2()