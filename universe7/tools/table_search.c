/* Universe 7 table search (host-side observer). Scores every K-subset of the given ops as a
 * prefix-coded instruction table by counting distinct final screens over all programs of 1..Lmax bits.
 *   gcc -O2 -fopenmp tools/table_search.c -o build/table_search
 *   ./build/table_search 13 127 4 "F1,F3,F4,F8,S3,C3,SKIP"      # Lmax steps K ops
 * See results/table-search.txt for the runs that chose the table in src/u7.hex. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
// op encoding: kind*64+k. kind 0=F+k (flip,move k) 1=M+k (move k) 2=S+k (set,move k) 3=C+k (clear,move k) 4=SKIP 5=SKIPN 6=LOOP 7=BACK 8=SKIP then move k? no.
static int STEPS=128;
static const char* name(int op,char*b){ int kd=op/64,k=op%64; const char*n[]={"F","M","S","C","SKIP","SKIPN","LOOP","BACK"}; if(kd<4) sprintf(b,"%s%d",n[kd],k); else sprintf(b,"%s",n[kd]); return b;}
static int tree[64][2];
static void build(const char**code,int*op,int n){ memset(tree,0,sizeof tree); int tn=1;
 for(int i=0;i<n;i++){ int k=0; const char*c=code[i]; for(int j=0;c[j];j++){ int b=c[j]-'0'; if(!c[j+1]) tree[k][b]=-(op[i]+1); else { if(tree[k][b]<=0) tree[k][b]=tn++; k=tree[k][b]; } } } }
static inline uint64_t run(uint64_t p,int L){
 uint64_t scr=0; int h=0,pc=0,skip=0;
 for(int s=0;s<STEPS;s++){ int k=0,op;
  for(;;){ int b=(p>>(L-1-pc))&1; pc=(pc+1)%L; int nx=tree[k][b]; if(nx<=0){op=-nx-1;break;} k=nx; }
  if(skip){skip=0;continue;}
  int kd=op/64,kk=op%64;
  switch(kd){ case 0: scr^=1ULL<<h; h=(h+kk)&63; break; case 1: h=(h+kk)&63; break; case 2: scr|=1ULL<<h; h=(h+kk)&63; break; case 3: scr&=~(1ULL<<h); h=(h+kk)&63; break;
   case 4: if(scr>>h&1) skip=1; break; case 5: if(!(scr>>h&1)) skip=1; break; case 6: if(scr>>h&1) pc=0; break; case 7: pc=0; break; }
 } return scr; }
static int cmp(const void*a,const void*b){uint64_t x=*(uint64_t*)a,y=*(uint64_t*)b;return x<y?-1:x>y;}
static long distinct(uint64_t*v,long n){ qsort(v,n,8,cmp); long d=n?1:0; for(long i=1;i<n;i++) if(v[i]!=v[i-1]) d++; return d; }
static long small[5]; static long score(int*op,int K,int Lmax,const char**code){ build(code,op,K); long t=0; for(int L=1;L<=Lmax;L++){ long n=1L<<L; uint64_t*S=malloc(n*8);
 #pragma omp parallel for schedule(static)
 for(long p=0;p<n;p++) S[p]=run(p,L); long d=distinct(S,n); if(L<=4) small[L]=d; t+=d; free(S);} return t; }
int main(int argc,char**argv){ int Lmax=atoi(argv[1]); STEPS=atoi(argv[2]); int K=atoi(argv[3]);
 int ops[200],no=0; char*spec=argv[4]; // comma list of op names
 for(char*t=strtok(spec,",");t;t=strtok(NULL,",")){ int kd=-1,k=0; if(!strcmp(t,"SKIP"))kd=4; else if(!strcmp(t,"SKIPN"))kd=5; else if(!strcmp(t,"LOOP"))kd=6; else if(!strcmp(t,"BACK"))kd=7; else { kd=t[0]=='F'?0:t[0]=='M'?1:t[0]=='S'?2:3; k=atoi(t+1);} ops[no++]=kd*64+k; }
 const char*c2[4]={"00","01","10","11"},*c3[3]={"0","10","11"},*c5[5]={"00","01","10","110","111"},*c8[8]={"000","001","010","011","100","101","110","111"};
 const char**code=K==4?c2:K==3?c3:K==5?c5:c8; int idx[8]; for(int i=0;i<K;i++) idx[i]=i; long totP=0; for(int L=1;L<=Lmax;L++) totP+=1L<<L;
 for(;;){ int op[8]; for(int i=0;i<K;i++) op[i]=ops[idx[i]]; long s=score(op,K,Lmax,code); char b[16];
  printf("%6.2f%% %ld [%ld %ld %ld %ld]",100.0*s/totP,s,small[1],small[2],small[3],small[4]); for(int i=0;i<K;i++) printf(" %s=%s",code[i],name(op[i],b)); printf("\n"); fflush(stdout);
  int i=K-1; while(i>=0&&idx[i]==no-K+i) i--; if(i<0)break; idx[i]++; for(int j=i+1;j<K;j++) idx[j]=idx[j-1]+1; } }
