import React, { createContext, useCallback, useReducer, type PropsWithChildren, type ReactNode } from 'react'


type userQueryAction = {
    action: 'user_query',
    value: string
}

type updateResponseQuery = {
    action: 'update_response',
    value: [number, string]
}

type updateTitle = {
    action: 'update_title',
    value: string
}

type Message = {
    role: 'user' | 'bot',
    content: string
}

type ChatState = {
    messages: Message[];
    chat_title: string;
};

type ChatContextType = ChatState & {
    askQuery: (query: string) => Promise<void>;
};

const ChatHistContext = createContext<ChatContextType>({
    messages: [],
    chat_title: '',
    askQuery: async (query: string) => undefined
})

const decoder = new TextDecoder();

async function* readStream(reader: ReadableStreamDefaultReader<Uint8Array<ArrayBufferLike>>) {
    let finished = false
    while (!finished) {
        let { value } = await reader.read();
        const { response, done } = JSON.parse(decoder.decode(value, { stream: true }))
        yield response;
        finished = done
    }
    return
}

const chathistReducer = (state: ChatState, payload: userQueryAction | updateResponseQuery | updateTitle): ChatState => {
    switch (payload.action) {
        case 'user_query': return {
            ...state,
            messages: [...state.messages, { role: 'user', content: payload.value }, { role: 'bot', content: '' }]

        };
        case 'update_response': return {
            ...state,
            messages: state.messages.map((el, index) => {
                if (index == payload.value[0]) {
                    return { role: 'bot', content: payload.value[1] }
                }
                return el;
            })
        }
        case 'update_title': return { ...state, chat_title: payload.value }
        default: return state;
    }
}

export const ChatHistContextProvider: React.FC<PropsWithChildren> = ({ children }) => {
    const [state, dispatch] = useReducer(chathistReducer, {
        messages: [],
        chat_title: 'New Title'
    } as ChatState)
    const askQuery = useCallback(async (query: string) => {
        let message = ''
        dispatch({ action: 'user_query', value: query });
        const response = await fetch('http://75.102.226.48:4000/api/v1/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ prompt: query })
        })
        if (response.body) {
            const reader = response.body.getReader()
            for await (const response of readStream(reader)) {
                message += response
                dispatch({ action: 'update_response', value: [state.messages.length + 1, message] })
            }
        }
        else {
            console.log('Error while streaming.')
        }
        if (state.chat_title === 'New Title') {
            getTitle(query = `User: ${query}\n Bot: ${message}`)
        }
        return undefined;
    }, [state.messages.length])

    const getTitle = useCallback(async (query: string) => {
        let message = ''
        const response = await fetch('http://75.102.226.48:4000/api/v1/title', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ prompt: query })
        })
        if (response.body) {
            const reader = response.body.getReader()
            for await (const response of readStream(reader)) {
                message += response
                dispatch({ action: 'update_title', value: message })
            }
        }
        else {
            console.log('Error while streaming.')
        }
        return undefined;
    }, [])

    return <ChatHistContext.Provider value={{ ...state, askQuery }}>
        {children}
    </ChatHistContext.Provider>
}

export default ChatHistContext;