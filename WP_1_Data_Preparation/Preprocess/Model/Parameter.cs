using System;
using System.Collections.Generic;
using Newtonsoft.Json;
using Newtonsoft.Json.Converters;

namespace Preprocess.Model
{
    public class Parameter
    {
        //MetaData
        public string Id { get; set; }
        public string InstanceId { get; set; }
        public string Name { get; set; }
        public string Nickname { get; set; }
        public string Description { get; set; }
        public int Index { get; set; }
        public ParameterType ParameterType { get; set; }

        //DataType
        public Type DataType { get; set; }
        public string TypeName { get; set; }
        public Access Access { get; set; }

        //State
        public DataMapping DataMapping { get; set; }
        public bool IsReversed { get; set; }
        public bool IsSimplified { get; set; }
        public bool IsLocked { get; set; }
        public bool IsNickNameMutable { get; set; }
        public WireDisplay WireDisplay { get; set; }

        //Data
        public object VolatileData { get; set; }
        public int VolatileDataCount { get; set; }

        //Connected
        public List<string> ConnectedIds { get; set; }
        public List<string> ConnectedInstanceIds { get; set; }
        public int ConnectedCount { get; set; }

    }

    [JsonConverter(typeof(StringEnumConverter))]
    public enum WireDisplay
    {
        //
        // Summary:
        //     Wire display is controlled by the application settings.
        @default,
        //
        // Summary:
        //     Wires are displayed faintly (thin and transparent) while the parameter is not
        //     selected.
        faint,
        //
        // Summary:
        //     Wires are not displayed at all while the parameter is not selected.
        hidden
    }
}

[JsonConverter(typeof(StringEnumConverter))]
public enum Access
{
    //
    // Summary:
    //     Every data item is to be treated individually.
    item,
    //
    // Summary:
    //     All data branches will be treated at the same time.
    list,
    //
    // Summary:
    //     The entire data structure will be treated at once.
    tree
}

[JsonConverter(typeof(StringEnumConverter))]
public enum DataMapping
{
    //
    // Summary:
    //     Data is not mapped.
    None,
    //
    // Summary:
    //     Data is flattened inside this parameter.
    Flatten,
    //
    // Summary:
    //     Data is grafted inside this parameter.
    Graft
}

[JsonConverter(typeof(StringEnumConverter))]
public enum ParameterType
{
    Input,
    Output
}

