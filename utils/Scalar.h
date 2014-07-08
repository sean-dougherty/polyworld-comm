#pragma once

#include <assert.h>
#include <iostream>

struct Scalar
{
	enum Type
	{
		INVALID,
		INT,
		FLOAT,
		BOOL
	};

	Scalar();
	Scalar( int ival );
	Scalar( long ival );
	Scalar( float fval );
	Scalar( double fval );
	Scalar( bool bval );
	Scalar( const Scalar &scalar );

	std::string str() const;

	operator int () const;
	operator long () const;
	operator float () const;
	operator double () const;
	operator bool () const;

	Type type;
	union {
		void *__val;
		int ival;
		float fval;
		bool bval;
	};
};

int operator + ( const Scalar &scalar, int i );
int operator + ( int i, const Scalar &scalar );
float operator - ( const Scalar &scalar, float f );
float operator - ( float f, const Scalar &scalar );
float operator * ( const Scalar &scalar, float f );
float operator * ( float f, const Scalar &scalar );
double operator * ( const Scalar &scalar, double f );
double operator * ( double f, const Scalar &scalar );
bool operator <= ( long i, const Scalar &scalar );
bool operator <= ( const Scalar &scalar, long i );
bool operator == ( const Scalar &scalar, double i );
bool operator == ( double i, const Scalar &scalar );
std::ostream &operator << ( std::ostream &, const Scalar & );

inline Scalar::operator int () const
{
	assert( type == INT );

	return ival;
}

inline Scalar::operator long () const
{
	return (int)*this;
}

inline Scalar::operator float () const
{
	assert( type == FLOAT );

	return fval;
}

inline Scalar::operator double () const
{
	return (float)*this;
}

inline Scalar::operator bool () const
{
	assert( type == BOOL );

	return bval;
}
