import React, { createContext, useCallback, useState, type PropsWithChildren } from 'react'

const ChatHistContext = createContext({
    messages: [],
    askQuery: async (query: string) => undefined
})


export const ChatHistContextProvider: React.FC<PropsWithChildren> = ({ children }) => {
    const [messages, setMessages] = useState([]);
    const askQuery = useCallback(async (query: string) => {
        const response = await fetch('http://127.0.0.1:4000/api/v1/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ prompt: query })
        })
        const reader = response.body?.getReader()
        console.log(reader)
        return undefined;
    }, [])
    return <ChatHistContext.Provider value={{ messages, askQuery }}>
        {children}
    </ChatHistContext.Provider>
}

export default ChatHistContext;