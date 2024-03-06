#출렵값: 삼각수중 500개 이상의 약수를 갖는 가장 작은 수
#풀이: 일단 1부터n까지의 자연수를 차례로 더해서 구해진 수가 삼각수이기 떄문에
# 반복문으로 삼각수를 구하면서 if문으로 약수개수를 확인후 도출
#약수는 1과 자신 뺴고 2부터 자신-1까지 수중 몇개 있냐

tri=0
for i in range(1,1000000):
    count=0
    tri+=i
    for j in range(1,int(tri**0.5)+1):
        if tri%j==0:
            count+=2
    if tri==j*j:
        count-=1       
    if count>499:
        print(tri)
        break
    
    