import { StyleSheet, Text, View } from "react-native";
import React, { useContext, useEffect } from "react";
import ChatHistContext from "@/app/_context/ChathistContext";
import Chat from "./Chat";
import { ScrollView } from "react-native-gesture-handler";

// This component is used to display the content of a chat conversation.
// It takes a UUID as a prop and uses the ChatHistContext to fetch messages related to that UUID.
// The messages are then displayed in a scrollable view, with each message styled according to its role (either 'user' or 'bot').

interface contentInterface {
    uuid: string
}
export default function Content({ uuid }: contentInterface) {
    const { messages, queryMessages } = useContext(ChatHistContext)
    useEffect(() => {
        const getMessages = async () =>
            await queryMessages(uuid)
        getMessages()
    }, [])
    return <ScrollView>
        {messages.map((msg, index) => {
            return <Chat key={index} role={msg.role}>
                <Text style={styles.contentStyle}>{msg.content}</Text>
            </Chat>
        })}
    </ScrollView>
}

const styles = StyleSheet.create({
    contentStyle: {
        fontSize: 15,
        lineHeight: 25,
        letterSpacing: 0.5
    },
});
