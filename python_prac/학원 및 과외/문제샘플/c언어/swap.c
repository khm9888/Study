/*
swap() is function which transpose the variable number value
by putting the address of "variable number a" to x,
and the address of "variable number b" to y.
Fill in the [[n]] and complete the program.
#Include order for standard function has been skipped.
*/
#include <stdio.h>

int main()
{
   int a = 123, b = 456;

   printf("a=%d b=%d\n", a, b);
   swap(&a, &b);
   printf("a=%d b=%d\n", a, b);

   return 0;
}

void swap(int *x, int *y)
{
   int temp;
   temp = *x;
   *x = *y;
   *y = temp;
}