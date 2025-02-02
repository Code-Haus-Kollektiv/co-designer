// Shared setup logic (runs only once)
using System;
using Codesigner;

public class TestFixture
{
    public GrasshopperAIClient Client { get; }

    public TestFixture()
    {
        Client = new GrasshopperAIClient(Environment.GetEnvironmentVariable("OPENAI_KEY"));
    }

}