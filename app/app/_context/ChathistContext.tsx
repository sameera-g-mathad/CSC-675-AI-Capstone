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

type updateMessages = {
    action: 'update_messages',
    value: { displayImage: string, chat_title: string, messages: any }
}

type Message = {
    role: 'user' | 'bot',
    content: string
}

type ChatState = {
    messages: Message[];
    chat_title: string;
    displayImage: string
};

type ChatContextType = ChatState & {
    askQuery: (uuid: string, query: string) => Promise<void>;
    queryMessages: (uuid: string) => Promise<void>;
};

const ChatHistContext = createContext<ChatContextType>({
    messages: [],
    chat_title: '',
    displayImage: '',
    askQuery: async (uuid: string, query: string) => undefined,
    queryMessages: async (uuid: string) => undefined
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

const chathistReducer = (state: ChatState, payload: userQueryAction | updateResponseQuery | updateTitle | updateMessages): ChatState => {
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
        case 'update_title': return { ...state, chat_title: payload.value };
        case 'update_messages': {
            const displayImage = payload.value['displayImage'] !== undefined ? payload.value['displayImage'] : ''
            return { ...state, displayImage, chat_title: payload.value['chat_title'], messages: [...state.messages, ...payload.value['messages']] }
        }
        default: return state;
    }
}

export const ChatHistContextProvider: React.FC<PropsWithChildren> = ({ children }) => {
    const [state, dispatch] = useReducer(chathistReducer, {
        messages: [],
        chat_title: 'New Title',
        displayImage: ''
    } as ChatState)

    const queryMessages = async (uuid: string) => {
        const response = await fetch('http://127.0.0.1:4000/api/v1/getMessages', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ uuid })
        })
        const json = await response.json();
        const chatDetails = json.data[0]
        const conversations = json.data['conversations']
        console.log(json)
        dispatch({ action: 'update_messages', value: { displayImage: chatDetails['displayImage'], chat_title: chatDetails['chat_title'], messages: conversations } })
    }


    const askQuery = useCallback(async (uuid: string, query: string) => {
        let message = ''
        dispatch({ action: 'user_query', value: query });
        const response = await fetch('http://127.0.0.1:4000/api/v1/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ prompt: query, uuid })
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
            getTitle(uuid, query = `User: ${query}\n Bot: ${message}`)
        }
        return undefined;
    }, [state.messages.length])

    const getTitle = useCallback(async (uuid: string, query: string) => {
        let message = ''
        const response = await fetch('http://127.0.0.1:4000/api/v1/title', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ prompt: query, uuid })
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
    return <ChatHistContext.Provider value={{ ...state, askQuery, queryMessages }}>
        {children}
    </ChatHistContext.Provider>
}

export default ChatHistContext;