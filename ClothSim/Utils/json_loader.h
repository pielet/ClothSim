#ifndef JSON_LOADER_H
#define JSON_LOADER_H

#include "../json/json.h"
#include <Eigen/Core>
#include "MathDef.h"

//! Used for enum type only
template <typename T>
void parse(T& value, const Json::Value& json)
{
	value = static_cast<T>(json.asInt());
}

template<>
void parse(bool& value, const Json::Value& json);

template<>
void parse(int& value, const Json::Value& json);

template<> 
void parse(float& value, const Json::Value& json);

template<> 
void parse(double& value, const Json::Value& json);

template <typename T1, typename T2>
void parse(T1& value, const Json::Value& json, const T2& default_value)
{
	if (json.isNull())
		value = static_cast<T1>(default_value);
	else
		parse<T1>(value, json);
}

template <typename T, int n>
void parse(cloth::Vec<T, n>& vec, const Json::Value& json)
{
	if (json.isNull())
		vec.setZero();
	else {
		for (int i = 0; i < n; ++i)
			parse<T>(vec(i), json[i]);
	}
}

template <typename T, int n>
void parse(Eigen::Matrix<T, n, 1>& vec, const Json::Value& json)
{
	if (json.isNull())
		vec.setZero();
	else {
		for (int i = 0; i < n; ++i)
			parse<T>(vec(i), json[i]);
	}
}

#endif // !JSON_LOADER_H
