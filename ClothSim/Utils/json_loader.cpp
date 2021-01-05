#include "json_loader.h"

template<>
void parse(float& value, const Json::Value& json)
{
	value = json.asFloat();
}

template<>
void parse(double& value, const Json::Value& json)
{
	value = json.asDouble();
}