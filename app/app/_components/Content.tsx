import { StyleSheet, Text, View } from "react-native";
import React, { useContext, useEffect } from "react";
import ChatHistContext from "@/app/_context/ChathistContext";
import Chat from "./Chat";
import { ScrollView } from "react-native-gesture-handler";

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
