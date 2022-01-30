import matplotlib.pyplot as plt
import matplotlib.axes as ax
                  


class BERT:
    def __init__(self):
        self.lines = []
        self.tap_weights = []
        self.FEC_codes = []
        
    def add_point(self, pre_FEC_BER, post_FEC_BER, tap_weights , FEC_code):
        
        added = False
            
        for line in self.lines:
            if line.FEC_code == FEC_code and line.tap_weights == tap_weights :
                line.add(pre_FEC_BER, post_FEC_BER)
                added = True
        
        if not added:
            newline = self.line(FEC_code, tap_weights)
            newline.add(pre_FEC_BER, post_FEC_BER)
            self.lines = self.lines + [newline]
            
    def plot(self):
        #plot = plt.figure()
        for line in self.lines:
            post_FEC = []
            pre_FEC = []
            for i in range (len(line.points)):
                #print(line.points[i])
                post_FEC = post_FEC + [line.points[i][0]]
                pre_FEC = pre_FEC + [line.points[i][1]]
                
            #print(post_FEC,pre_FEC)
            #plt.loglog(post_FEC,pre_FEC)
            plt.loglog(pre_FEC,post_FEC, "b*" ,label = "RS(544,536,4), h = [0.6, 0.2, -0.2]")
       
        plt.xlabel("pre-FEC BER")
        plt.ylabel("post-FEC BER")
        plt.grid()
        
        #plt.xlim([1e-2, 1e-7])
        #plt.ylim([1e-5, 1e-2])
        plt.show()
            
    class line:
        def __init__(self, FEC_code, tap_weights):
            self.FEC_code = FEC_code
            self.tap_weights = tap_weights
            self.points = []
            
        def add(self, pre_FEC_BER, post_FEC_BER):
            #print(0)
            
            if len(self.points) == 0:
                #print(1)
                self.points = [[post_FEC_BER, pre_FEC_BER]]
                return True
            
            if self.points[0][0] < post_FEC_BER:
                #print(2)
                self.points =  [[post_FEC_BER, pre_FEC_BER]] + self.points
                return True
                
            for point_idx in range(len(self.points)):
                if self.points[point_idx][0] < post_FEC_BER:
                    #print(3,point_idx)
                    self.points =  self.points[:point_idx] + [[post_FEC_BER, pre_FEC_BER]] + self.points[point_idx:]
                    return True
            
            #print(3)
            self.points = self.points + [[post_FEC_BER, pre_FEC_BER]]
            return True

#%%


bert = BERT()


tap_weights = '[0.6, 0.2, -0.2]'
FEC_code = 'RS(544,536)'



bert.add_point(0.03415591078931959, 0.034794674702931586, tap_weights , FEC_code)

bert.add_point(0.027440123443838966, 0.027348414045661754 ,tap_weights , FEC_code)

bert.add_point(0.02053169351900772, 0.020192069274638083 ,tap_weights , FEC_code)

bert.add_point(0.014490155201254275, 0.014204924755383472 ,tap_weights , FEC_code)

bert.add_point(0.008613602854924879,0.008452383223 ,tap_weights , FEC_code)

bert.add_point(0.00419712867189154, 0.004249556543134525 ,tap_weights , FEC_code)

bert.add_point(0.001519206083690803,  0.0013389536325316143, tap_weights , FEC_code)

bert.add_point(0.0002851491644843378,  2.1076121993553185e-05 ,tap_weights , FEC_code)

bert.add_point( 0.00023078962476658776, 1.1126157915148741e-05 ,tap_weights , FEC_code)

bert.add_point( 0.0001759532469811382, 7.667512254668218e-06 ,tap_weights , FEC_code)

bert.add_point( 0.00013160730668507897, 5.5040422012899074e-06 ,tap_weights , FEC_code)

bert.add_point( 9.568550558504534e-05, 2.214269851093641e-06 ,tap_weights , FEC_code)

bert.add_point( 7.05720340195351e-05,  7.257714354314462e-07 ,tap_weights , FEC_code)

bert.add_point( 0.0012455010328312546, 0.001026149650002861 ,tap_weights , FEC_code)

bert.add_point(  0.0007820144310272809,0.0003814682713765283 ,tap_weights , FEC_code)

bert.add_point(  0.0004024513291384299, 0.00010013542123633867 ,tap_weights , FEC_code)

bert.plot()
#%%
#tap_weights = [1,0.55,0.3]

#FEC_code = 'KR4'

#0.1 Noise Varience

#NO FEC
#Bits Transmitted = 10485740 Bit Errors = 9970
#Bit Error Ratio =  0.0009508151069929256

#FEC
#Bits Transmitted = 10485740 Bit Errors = 1118
#Bit Error Ratio =  0.00010662099193762195

#bert.add_point( 0.0009508151069929256, 0.00010662099193762195 ,tap_weights , FEC_code)

#0.09 Noise Varience

#NO FEC
#Bits Transmitted = 10485740 Bit Errors = 5353
#Bit Error Ratio =  0.0005105028352791506

#FEC
#Bits Transmitted = 37748880 Bit Errors = 245
#Bit Error Ratio =  6.490258783836766e-06


#bert.add_point( 0.0005105028352791506, 6.490258783836766e-06 ,tap_weights , FEC_code)


#tst.plot()