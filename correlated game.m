function P=ce(w,U1,U2,sign)
% P = CE(w,U1,U2) finds efficient/inefficient correlated equilibria that
% maximize weighted sum of the payoffs in a two player normal form game
%     max sign*(w*U1(P)+(1-w)*U2(P))
%    s.t. P is correlated equilibrium
%
% INPUT:  w is a weight vector for player 1 of length k
%         U1 is n-by-m payoff matrix of player 1
%         U2 is n-by-m payoff matrix of player 2
% OUTPUT: P is k-by-n*m matrix of correlated equilibrium strategies
%
% Algorithm: builds the linear inequalities that represent the rationality
% constraints for two player. The constraint matrix A is constructed so that
% if P is the probability distribution over joint actions, and if X=P(:),
% the correlated equilibrium constraints are A * X <= 0.


if any(size(U1)~=size(U2))
    error('U1 and U2 must be of the same size')
end
[n,m]=size(U1);
A1=zeros(n*(n-1),n*m);
for i=1:n
    A1((i-1)*(n-1)+(1:n-1),i:n:end)=U1([1:i-1 i+1:n],:)-repmat(U1(i,:),n-1,1);
end
A2=zeros(m*(m-1),n*m);
for i=1:m
    A2((i-1)*(m-1)+(1:m-1),(i-1)*n+(1:n))=(U2(:,[1:i-1 i+1:m])-repmat(U2(:,i),1,m-1))';
end
A=[A1;A2];
b=zeros(n*(n-1)+m*(m-1),1);
Aeq=ones(1,n*m);
beq=1;
lb=zeros(n*m,1);
ub=ones(n*m,1);
o=optimset('Display','off');
k=length(w);
P=nan(k,n*m);
for i=1:k
    f=-sign*(w(i)*U1(:)+(1-w(i))*U2(:));
    P(i,:)=linprog(f,A,b,Aeq,beq,lb,ub,[],o)';
end

% Example 1: Prisoner's dilemma
u1=[6 1
    5 0]; % payoff of player 1
u2=[0 1
    5 6]; % payoff of player 2
ne=[1 0 0 1]; % Nash equilibria
plotce(u1,u2,ne,'PRISONER''S DILEMMA')

% Example 2: Battle of sexes
u1=[3 0
    0 1]; % payoff of player 1
u2=[1 0
    0 3]; % payoff of player 2
ne=[[3 1 1 3]/4
     0 1 0 1
     1 0 1 0]; % Nash equilibria
plotce(u1,u2,ne,'GAME OF BATTLE OF SEXES')

% Example 3: Game of chicken (Aumann, 1974)
u1=[5 0
    4 1]; % payoff of player 1
u2=[1 0
    4 5]; % payoff of player 2
ne=[[1 1 1 1]/2
     0 1 0 1
     1 0 1 0]; % Nash equilibria
plotce(u1,u2,ne,'GAME OF CHICKEN (AUMANN, 1974)')
