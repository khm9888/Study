/*
keisan() is the function to execute four arithmetic operations,
and it use pointer to the function as a parameter.
Fill in the [[n]] and complete the program which find the product of "100" and "20".
#Include order for standard function has been skipped.
*/

void keisan(void (*calc)(int a, int b), int dt1, int dt2);
void wa(int a, int b);
void sa(int a, int b);
void seki(int a, int b);
void shou(int a, int b);

int main()
{
	keisan( seki , 100 , 20  );
	return 0;
}

void keisan(void (*calc)(int a, int b), int dt1, int dt2)
{
	calc(dt1, dt2);
}

void wa(int a, int b)
{
	printf("%d\n",a + b);
}

void sa(int a, int b)
{
	printf("%d\n",a - b);
}

void seki(int a, int b)
{
	printf("%d\n",a * b);
}

void shou(int a, int b)
{
	printf("%d\n",a / b);
}

